from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class DefaultControlComponentFormatTests(tests.TestCase):
    """Tests for default ControlComponentFormat implementation."""

    def test_check_support_status_unsupported(self):
        self.assertRaises(errors.UnsupportedFormatError, UnsupportedControlComponentFormat().check_support_status, allow_unsupported=False)
        UnsupportedControlComponentFormat().check_support_status(allow_unsupported=True)

    def test_check_support_status_supported(self):
        controldir.ControlComponentFormat().check_support_status(allow_unsupported=False)
        controldir.ControlComponentFormat().check_support_status(allow_unsupported=True)

    def test_recommend_upgrade_current_format(self):
        ui.ui_factory = tests.TestUIFactory()
        format = controldir.ControlComponentFormat()
        format.check_support_status(allow_unsupported=False, recommend_upgrade=True)
        self.assertEqual('', ui.ui_factory.stderr.getvalue())

    def test_recommend_upgrade_old_format(self):
        ui.ui_factory = tests.TestUIFactory()
        format = OldControlComponentFormat()
        format.check_support_status(allow_unsupported=False, recommend_upgrade=False)
        self.assertEqual('', ui.ui_factory.stderr.getvalue())
        format.check_support_status(allow_unsupported=False, recommend_upgrade=True, basedir='apath')
        self.assertEqual('An old format that is slow is deprecated and a better format is available.\nIt is recommended that you upgrade by running the command\n  brz upgrade apath\n', ui.ui_factory.stderr.getvalue())