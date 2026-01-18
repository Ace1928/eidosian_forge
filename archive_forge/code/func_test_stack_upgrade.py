from .. import check, controldir, errors, tests
from ..upgrade import upgrade
from .scenarios import load_tests_apply_scenarios
def test_stack_upgrade(self):
    """Correct checks when stacked-on repository is upgraded.

        We initially stack on a repo with the same rich root support,
        we then upgrade it and should fail, we then upgrade the overlaid
        repository.
        """
    base = self.make_branch_and_tree('base', format=self.scenario_old_format)
    self.build_tree(['base/foo'])
    base.commit('base commit')
    stacked = base.controldir.sprout('stacked', stacked=True)
    self.assertTrue(stacked.open_branch().get_stacked_on_url())
    new_format = controldir.format_registry.make_controldir(self.scenario_new_format)
    upgrade('base', new_format)
    if self.scenario_model_change:
        self.assertRaises(errors.IncompatibleRepositories, stacked.open_branch)
    else:
        check.check_dwim('stacked', False, True, True)
    stacked = controldir.ControlDir.open('stacked')
    upgrade('stacked', new_format)
    stacked = controldir.ControlDir.open('stacked')
    check.check_dwim('stacked', False, True, True)