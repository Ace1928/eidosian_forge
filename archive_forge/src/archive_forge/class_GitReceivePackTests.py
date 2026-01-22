import shutil
import tempfile
from dulwich.tests import BlackboxTestCase
from ..repo import Repo
class GitReceivePackTests(BlackboxTestCase):
    """Blackbox tests for dul-receive-pack."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)

    def test_basic(self):
        process = self.run_command('dul-receive-pack', [self.path])
        stdout, stderr = process.communicate(b'0000')
        self.assertEqual(b'0000', stdout[-4:])
        self.assertEqual(0, process.returncode)

    def test_missing_arg(self):
        process = self.run_command('dul-receive-pack', [])
        stdout, stderr = process.communicate()
        self.assertEqual([b'usage: dul-receive-pack <git-dir>'], stderr.splitlines()[-1:])
        self.assertEqual(b'', stdout)
        self.assertEqual(1, process.returncode)