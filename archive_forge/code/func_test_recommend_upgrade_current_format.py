from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_recommend_upgrade_current_format(self):
    ui.ui_factory = tests.TestUIFactory()
    format = controldir.ControlComponentFormat()
    format.check_support_status(allow_unsupported=False, recommend_upgrade=True)
    self.assertEqual('', ui.ui_factory.stderr.getvalue())