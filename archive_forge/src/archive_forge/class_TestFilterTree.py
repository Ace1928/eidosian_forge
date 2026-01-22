import tarfile
import zipfile
from .. import export, filter_tree, tests
from . import fixtures
from .test_filters import _stack_1
class TestFilterTree(tests.TestCaseWithTransport):

    def make_tree(self):
        self.underlying_tree = fixtures.make_branch_and_populated_tree(self)

        def stack_callback(path):
            return _stack_1
        self.filter_tree = filter_tree.ContentFilterTree(self.underlying_tree, stack_callback)
        return self.filter_tree

    def test_get_file_text(self):
        self.make_tree()
        self.assertEqual(self.underlying_tree.get_file_text('hello'), b'hello world')
        self.assertEqual(self.filter_tree.get_file_text('hello'), b'HELLO WORLD')

    def test_tar_export_content_filter_tree(self):
        self.make_tree()
        export.export(self.filter_tree, 'out.tgz')
        ball = tarfile.open('out.tgz', 'r:gz')
        self.assertEqual(b'HELLO WORLD', ball.extractfile('out/hello').read())

    def test_zip_export_content_filter_tree(self):
        self.make_tree()
        export.export(self.filter_tree, 'out.zip')
        zipf = zipfile.ZipFile('out.zip', 'r')
        self.assertEqual(b'HELLO WORLD', zipf.read('out/hello'))