import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
class NonlinearConvex_OverlappingCircles(unittest.TestCase):

    def check_cuts_valid_for_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1)
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(True)
        m.lower_circle.indicator_var.fix(False)
        m.upper_circle2.indicator_var.fix(True)
        m.lower_circle2.indicator_var.fix(False)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    def check_cuts_valid_on_facet_containing_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1)
        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(False)
        m.lower_circle.indicator_var.fix(True)
        m.upper_circle2.indicator_var.fix(False)
        m.lower_circle2.indicator_var.fix(True)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tightM_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tightM_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_tightM_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_tightM_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)