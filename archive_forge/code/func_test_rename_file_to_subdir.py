from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_rename_file_to_subdir(self):
    tree = self.make_branch_and_memory_tree('branch')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add('')
    tree.mkdir('subdir', b'subdir-id')
    tree.add('foo', 'file', b'foo-id')
    tree.put_file_bytes_non_atomic('foo', b'content\n')
    tree.commit('one', rev_id=b'rev-one')
    tree.rename_one('foo', 'subdir/bar')
    self.assertEqual('subdir/bar', tree.id2path(b'foo-id'))
    self.assertEqual(b'content\n', tree._file_transport.get_bytes('subdir/bar'))
    tree.commit('two', rev_id=b'rev-two')
    rev_tree2 = tree.branch.repository.revision_tree(b'rev-two')
    self.assertEqual('subdir/bar', rev_tree2.id2path(b'foo-id'))