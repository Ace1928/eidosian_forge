from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_put_new_file(self):
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    with tree.lock_write():
        tree.add(['', 'foo'], kinds=['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo', b'barshoom')
        self.assertEqual(b'barshoom', tree.get_file('foo').read())