from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestCanonicalPaths(TestCaseWithWorkingTree):

    def _make_canonical_test_tree(self, commit=True):
        work_tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/dir/', 'tree/dir/file'])
        work_tree.add(['dir', 'dir/file'])
        if commit:
            work_tree.commit('commit 1')
        return work_tree

    def test_canonical_path(self):
        work_tree = self._make_canonical_test_tree()
        if features.CaseInsensitiveFilesystemFeature.available():
            self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
        elif features.CaseInsCasePresFilenameFeature.available():
            self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
        else:
            self.assertEqual('Dir/File', work_tree.get_canonical_path('Dir/File'))

    def test_canonical_path_before_commit(self):
        work_tree = self._make_canonical_test_tree(False)
        if features.CaseInsensitiveFilesystemFeature.available():
            self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
        elif features.CaseInsCasePresFilenameFeature.available():
            self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
        else:
            self.assertEqual('Dir/File', work_tree.get_canonical_path('Dir/File'))

    def test_canonical_path_dir(self):
        work_tree = self._make_canonical_test_tree()
        if features.CaseInsensitiveFilesystemFeature.available():
            self.assertEqual('dir', work_tree.get_canonical_path('Dir'))
        elif features.CaseInsCasePresFilenameFeature.available():
            self.assertEqual('dir', work_tree.get_canonical_path('Dir'))
        else:
            self.assertEqual('Dir', work_tree.get_canonical_path('Dir'))

    def test_canonical_path_root(self):
        work_tree = self._make_canonical_test_tree()
        self.assertEqual('', work_tree.get_canonical_path(''))
        self.assertEqual('', work_tree.get_canonical_path('/'))

    def test_canonical_path_invalid_all(self):
        work_tree = self._make_canonical_test_tree()
        self.assertEqual('foo/bar', work_tree.get_canonical_path('foo/bar'))

    def test_canonical_invalid_child(self):
        work_tree = self._make_canonical_test_tree()
        if features.CaseInsensitiveFilesystemFeature.available():
            self.assertEqual('dir/None', work_tree.get_canonical_path('Dir/None'))
        elif features.CaseInsCasePresFilenameFeature.available():
            self.assertEqual('dir/None', work_tree.get_canonical_path('Dir/None'))
        else:
            self.assertEqual('Dir/None', work_tree.get_canonical_path('Dir/None'))

    def test_canonical_tree_name_mismatch(self):
        self.requireFeature(features.case_sensitive_filesystem_feature)
        work_tree = self.make_branch_and_tree('.')
        self.build_tree(['test/', 'test/file', 'Test'])
        work_tree.add(['test/', 'test/file', 'Test'])
        self.assertEqual(['test', 'Test', 'test/file', 'Test/file'], list(work_tree.get_canonical_paths(['test', 'Test', 'test/file', 'Test/file'])))
        test_revid = work_tree.commit('commit')
        test_tree = work_tree.branch.repository.revision_tree(test_revid)
        test_tree.lock_read()
        self.addCleanup(test_tree.unlock)
        self.assertEqual(['', 'Test', 'test', 'test/file'], [p for p, e in test_tree.iter_entries_by_dir()])