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
def test_remove_no_matrix(self):
    m = pyo.ConcreteModel()
    m.v1 = pyo.Var()
    igraph = IncidenceGraphInterface()
    with self.assertRaisesRegex(RuntimeError, 'no incidence matrix'):
        igraph.remove_nodes([m.v1])