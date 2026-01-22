from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestPutFileBytesNonAtomic(TestCaseWithWorkingTree):

    def test_put_new_file(self):
        t = self.make_branch_and_tree('t1')
        t.add(['foo'], kinds=['file'])
        t.put_file_bytes_non_atomic('foo', b'barshoom')
        with t.get_file('foo') as f:
            self.assertEqual(b'barshoom', f.read())

    def test_put_existing_file(self):
        t = self.make_branch_and_tree('t1')
        t.add(['foo'], kinds=['file'])
        t.put_file_bytes_non_atomic('foo', b'first-content')
        t.put_file_bytes_non_atomic('foo', b'barshoom')
        with t.get_file('foo') as f:
            self.assertEqual(b'barshoom', f.read())