from breezy import errors
from breezy.tests.per_controldir import TestCaseWithControlDir
class TestControlDir(TestCaseWithControlDir):

    def test_get_format_description(self):
        self.assertIsInstance(self.bzrdir_format.get_format_description(), str)

    def test_is_supported(self):
        self.assertIsInstance(self.bzrdir_format.is_supported(), bool)

    def test_upgrade_recommended(self):
        self.assertIsInstance(self.bzrdir_format.upgrade_recommended, bool)

    def test_supports_transport(self):
        self.assertIsInstance(self.bzrdir_format.supports_transport(self.get_transport()), bool)

    def test_check_support_status(self):
        if not self.bzrdir_format.is_supported():
            self.assertRaises((errors.UnsupportedFormatError, errors.UnsupportedVcs), self.bzrdir_format.check_support_status, False)
        else:
            self.bzrdir_format.check_support_status(True)
            self.bzrdir_format.check_support_status(False)