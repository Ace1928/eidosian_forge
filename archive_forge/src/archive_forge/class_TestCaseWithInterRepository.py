from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
class TestCaseWithInterRepository(TestCaseWithControlDir):

    def setUp(self):
        super().setUp()
        if self.extra_setup:
            self.extra_setup(self)

    def get_default_format(self):
        self.assertEqual(self.repository_format._matchingcontroldir.repository_format, self.repository_format)
        return self.repository_format._matchingcontroldir

    def make_branch(self, relpath, format=None):
        repo = self.make_repository(relpath, format=format)
        return repo.controldir.create_branch()

    def make_controldir(self, relpath, format=None):
        try:
            url = self.get_url(relpath)
            segments = url.split('/')
            if segments and segments[-1] not in ('', '.'):
                parent = '/'.join(segments[:-1])
                t = transport.get_transport(parent)
                try:
                    t.mkdir(segments[-1])
                except FileExists:
                    pass
            if format is None:
                format = self.repository_format._matchingcontroldir
            return format.initialize(url)
        except UninitializableFormat:
            raise TestSkipped('Format %s is not initializable.' % format)

    def make_repository(self, relpath, format=None):
        made_control = self.make_controldir(relpath, format=format)
        return self.repository_format.initialize(made_control)

    def make_to_repository(self, relpath):
        made_control = self.make_controldir(relpath, self.repository_format_to._matchingcontroldir)
        return self.repository_format_to.initialize(made_control)