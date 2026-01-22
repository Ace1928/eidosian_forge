import shutil
import tempfile
from dulwich.tests import BlackboxTestCase
from ..repo import Repo
class GitUploadPackTests(BlackboxTestCase):
    """Blackbox tests for dul-upload-pack."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)

    def test_missing_arg(self):
        process = self.run_command('dul-upload-pack', [])
        stdout, stderr = process.communicate()
        self.assertEqual([b'usage: dul-upload-pack <git-dir>'], stderr.splitlines()[-1:])
        self.assertEqual(b'', stdout)
        self.assertEqual(1, process.returncode)