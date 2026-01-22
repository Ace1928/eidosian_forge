from breezy.controldir import ControlDirFormat
from breezy.tests import (TestCaseWithTransport, default_transport,
from breezy.transport import memory
class TestCaseWithControlDir(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.controldir = None

    def get_bzrdir(self):
        if self.controldir is None:
            self.controldir = self.make_controldir(None)
        return self.controldir

    def get_default_format(self):
        return self.bzrdir_format