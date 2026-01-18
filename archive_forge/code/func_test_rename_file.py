from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_rename_file(self):
    tree = self.make_branch_and_memory_tree('branch')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add(['', 'foo'], ['directory', 'file'], ids=[b'root-id', b'foo-id'])
    tree.put_file_bytes_non_atomic('foo', b'content\n')
    tree.commit('one', rev_id=b'rev-one')
    tree.rename_one('foo', 'bar')
    self.assertEqual('bar', tree.id2path(b'foo-id'))
    self.assertEqual(b'content\n', tree._file_transport.get_bytes('bar'))
    self.assertRaises(NoSuchFile, tree._file_transport.get_bytes, 'foo')
    tree.commit('two', rev_id=b'rev-two')
    self.assertEqual(b'content\n', tree._file_transport.get_bytes('bar'))
    self.assertRaises(NoSuchFile, tree._file_transport.get_bytes, 'foo')
    rev_tree2 = tree.branch.repository.revision_tree(b'rev-two')
    self.assertEqual('bar', rev_tree2.id2path(b'foo-id'))
    self.assertEqual(b'content\n', rev_tree2.get_file_text('bar'))