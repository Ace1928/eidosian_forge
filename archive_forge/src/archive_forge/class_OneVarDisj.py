import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
class OneVarDisj(unittest.TestCase):

    def check_no_cuts_for_optimal_m(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 0)

    def check_expected_two_segment_cut(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 1)
        cut_expr = cuts[0].body
        m.x.fix(0)
        m.disj1.indicator_var.fix(True)
        m.disj2.indicator_var.fix(False)
        self.assertAlmostEqual(value(cut_expr), 0)
        m.x.fix(2)
        m.disj2.indicator_var.fix(True)
        m.disj1.indicator_var.fix(False)
        self.assertAlmostEqual(value(cut_expr), 0)

    def check_two_segment_cuts_valid(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            cut_lower = cut.lower
            cut_upper = cut.upper
            m.x.fix(0)
            m.disj2.indicator_var.fix(False)
            check_validity(self, cut_expr, cut_lower, cut_upper, TOL=1e-08)
            m.x.fix(1)
            check_validity(self, cut_expr, cut_lower, cut_upper)
            m.x.fix(2)
            m.disj2.indicator_var.fix(True)
            check_validity(self, cut_expr, cut_lower, cut_upper)
            m.x.fix(3)
            check_validity(self, cut_expr, cut_lower, cut_upper)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_no_cuts_for_optimal_m(self):
        m = models.oneVarDisj_2pts()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_no_cuts_for_optimal_m_fme(self):
        m = models.oneVarDisj_2pts()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_no_cuts_for_optimal_m_inf_norm(self):
        m = models.oneVarDisj_2pts()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'), post_process_cut=None)
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_expected_two_segment_cut(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, verbose=True)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_expected_two_segment_cut_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_expected_two_segment_cut_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()
        m.dual = Suffix(direction=Suffix.IMPORT)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'), post_process_cut=None)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_expected_two_segment_cut_inf_norm_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'), create_cuts=create_cuts_fme, post_process_cut=None, verbose=True)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_deactivated_objectives_ignored(self):
        m = models.twoSegments_SawayaGrossmann()
        m.another_obj = Objective(expr=m.x - m.disj2.indicator_var, sense=maximize)
        m.another_obj.deactivate()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, verbose=True)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_two_segment_cuts_valid(self):
        m = models.twoSegments_SawayaGrossmann()
        m.will_be_stale = Var()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_two_segment_cuts_valid(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_two_segment_cuts_valid_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_two_segment_cuts_valid(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_two_segment_cuts_valid_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()
        m.dual = Var()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'))
        self.check_two_segment_cuts_valid(m)

    def check_expected_two_segment_cut_exact(self, cuts):
        m = cuts.model()
        self.assertEqual(len(cuts), 1)
        cut_expr = cuts[0].body
        m.x.fix(0)
        m.disj1.indicator_var.fix(True)
        m.disj2.indicator_var.fix(False)
        self.assertEqual(value(cut_expr), 0)
        m.x.fix(2)
        m.disj2.indicator_var.fix(True)
        m.disj1.indicator_var.fix(False)
        self.assertEqual(value(cut_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_integer_arithmetic_cuts_valid_l2(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_integer_arithmetic_cuts_valid_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, norm=float('inf'), post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_maximization(self):
        m = models.twoSegments_SawayaGrossmann()
        m.obj.expr = -m.obj.expr
        m.obj.sense = maximize
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_named_correctly(self):
        m = models.twoSegments_SawayaGrossmann()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, cuts_name='perfect_cuts', post_process_cut=None, do_integer_arithmetic=True)
        cuts = m.component('perfect_cuts')
        self.assertIsInstance(cuts, Constraint)
        self.assertIsNone(m._pyomo_gdp_cuttingplane_transformation.component('cuts'))
        self.check_expected_two_segment_cut_exact(cuts)

    def test_non_unique_cut_name_error(self):
        m = models.twoSegments_SawayaGrossmann()
        self.assertRaisesRegex(GDP_Error, "cuts_name was specified as 'disj1', but this is already a component on the instance! Please specify a unique name.", TransformationFactory('gdp.cuttingplane').apply_to, m, cuts_name='disj1')