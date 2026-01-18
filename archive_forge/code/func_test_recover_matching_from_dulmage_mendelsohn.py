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
def test_recover_matching_from_dulmage_mendelsohn(self):
    m = make_degenerate_solid_phase_model()
    igraph = IncidenceGraphInterface(m)
    vdmp, cdmp = igraph.dulmage_mendelsohn()
    vmatch = vdmp.underconstrained + vdmp.square + vdmp.overconstrained
    cmatch = cdmp.underconstrained + cdmp.square + cdmp.overconstrained
    self.assertEqual(len(ComponentSet(vmatch)), len(vmatch))
    self.assertEqual(len(ComponentSet(cmatch)), len(cmatch))
    matching = list(zip(vmatch, cmatch))
    for var, con in matching:
        var_in_con = ComponentSet(igraph.get_adjacent_to(con))
        self.assertIn(var, var_in_con)