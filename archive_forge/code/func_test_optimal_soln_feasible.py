from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
@unittest.skipIf('gurobi' not in linear_solvers, 'Gurobi solver not available')
def test_optimal_soln_feasible(self):
    m = ConcreteModel()
    m.Points = RangeSet(3)
    m.Centroids = RangeSet(2)
    m.X = Param(m.Points, initialize={1: 0.3672, 2: 0.8043, 3: 0.3059})
    m.cluster_center = Var(m.Centroids, bounds=(0, 2))
    m.distance = Var(m.Points, bounds=(0, 2))
    m.t = Var(m.Points, m.Centroids, bounds=(0, 2))

    @m.Disjunct(m.Points, m.Centroids)
    def AssignPoint(d, i, k):
        m = d.model()
        d.LocalVars = Suffix(direction=Suffix.LOCAL)
        d.LocalVars[d] = [m.t[i, k]]

        def distance1(d):
            return m.t[i, k] >= m.X[i] - m.cluster_center[k]

        def distance2(d):
            return m.t[i, k] >= -(m.X[i] - m.cluster_center[k])
        d.dist1 = Constraint(rule=distance1)
        d.dist2 = Constraint(rule=distance2)
        d.define_distance = Constraint(expr=m.distance[i] == m.t[i, k])

    @m.Disjunction(m.Points)
    def OneCentroidPerPt(m, i):
        return [m.AssignPoint[i, k] for k in m.Centroids]
    m.obj = Objective(expr=sum((m.distance[i] for i in m.Points)))
    TransformationFactory('gdp.hull').apply_to(m)
    m.AssignPoint[1, 1].indicator_var.fix(True)
    m.AssignPoint[1, 2].indicator_var.fix(False)
    m.AssignPoint[2, 1].indicator_var.fix(False)
    m.AssignPoint[2, 2].indicator_var.fix(True)
    m.AssignPoint[3, 1].indicator_var.fix(True)
    m.AssignPoint[3, 2].indicator_var.fix(False)
    m.cluster_center[1].fix(0.3059)
    m.cluster_center[2].fix(0.8043)
    m.distance[1].fix(0.0613)
    m.distance[2].fix(0)
    m.distance[3].fix(0)
    m.t[1, 1].fix(0.0613)
    m.t[1, 2].fix(0)
    m.t[2, 1].fix(0)
    m.t[2, 2].fix(0)
    m.t[3, 1].fix(0)
    m.t[3, 2].fix(0)
    results = SolverFactory('gurobi').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    TOL = 1e-08
    for c in m.component_data_objects(Constraint, active=True):
        if c.lower is not None:
            self.assertGreaterEqual(value(c.body) + TOL, value(c.lower))
        if c.upper is not None:
            self.assertLessEqual(value(c.body) - TOL, value(c.upper))