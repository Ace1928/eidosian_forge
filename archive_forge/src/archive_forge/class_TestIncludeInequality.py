import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(networkx_available, 'networkx is not available.')
@unittest.skipUnless(scipy_available, 'scipy is not available.')
class TestIncludeInequality(unittest.TestCase):

    def make_model_with_inequalities(self):
        m = make_degenerate_solid_phase_model()

        @m.Constraint()
        def flow_bound(m):
            return m.flow >= 0

        @m.Constraint(m.components)
        def flow_comp_bound(m, j):
            return m.flow_comp[j] >= 0
        return m

    def test_dont_include_inequality_model(self):
        m = self.make_model_with_inequalities()
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        self.assertEqual(igraph.incidence_matrix.shape, (8, 8))

    def test_include_inequality_model(self):
        m = self.make_model_with_inequalities()
        igraph = IncidenceGraphInterface(m, include_inequality=True)
        self.assertEqual(igraph.incidence_matrix.shape, (12, 8))

    @unittest.skipUnless(asl_available, 'pynumero_ASL is not available')
    def test_dont_include_inequality_nlp(self):
        m = self.make_model_with_inequalities()
        m._obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(m)
        igraph = IncidenceGraphInterface(nlp, include_inequality=False)
        self.assertEqual(igraph.incidence_matrix.shape, (8, 8))

    @unittest.skipUnless(asl_available, 'pynumero_ASL is not available')
    def test_include_inequality_nlp(self):
        m = self.make_model_with_inequalities()
        m._obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(m)
        igraph = IncidenceGraphInterface(nlp, include_inequality=True)
        self.assertEqual(igraph.incidence_matrix.shape, (12, 8))