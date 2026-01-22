import shutil
import tempfile
import unittest
class QueuelibTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='queuelib-tests-')
        self.qpath = self.tempfilename()
        self.qdir = self.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def tempfilename(self):
        with tempfile.NamedTemporaryFile(dir=self.tmpdir) as nf:
            return nf.name

    def mkdtemp(self):
        return tempfile.mkdtemp(dir=self.tmpdir)