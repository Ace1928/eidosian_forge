from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
class TestChangeLogMerger(tests.TestCaseWithTransport):
    """Tests for ChangeLogMerger class.

    Most tests should be unit tests for merge_entries (and its helpers).
    This class is just to cover the handful of lines of code in ChangeLogMerger
    itself.
    """

    def make_builder(self):
        builder = test_merge_core.MergeBuilder(self.test_base_dir)
        self.addCleanup(builder.cleanup)
        return builder

    def make_changelog_merger(self, base_text, this_text, other_text):
        builder = self.make_builder()
        clog = builder.add_file(builder.root(), 'ChangeLog', base_text, True, file_id=b'clog-id')
        builder.change_contents(clog, other=other_text, this=this_text)
        merger = builder.make_merger(merge.Merge3Merger, ['ChangeLog'])
        merger.this_branch.get_config().set_user_option('changelog_merge_files', 'ChangeLog')
        merge_hook_params = merge.MergeFileHookParams(merger, ['ChangeLog', 'ChangeLog', 'ChangeLog'], None, 'file', 'file', 'conflict')
        changelog_merger = changelog_merge.ChangeLogMerger(merger)
        return (changelog_merger, merge_hook_params)

    def test_merge_text_returns_not_applicable(self):
        """A conflict this plugin cannot resolve returns (not_applicable, None).
        """

        def entries_as_str(entries):
            return b''.join((entry + b'\n' for entry in entries))
        changelog_merger, merge_hook_params = self.make_changelog_merger(entries_as_str(sample2_base_entries), b'', entries_as_str(sample2_other_entries))
        self.assertEqual(('not_applicable', None), changelog_merger.merge_contents(merge_hook_params))

    def test_merge_text_returns_success(self):
        """A successful merge returns ('success', lines)."""
        changelog_merger, merge_hook_params = self.make_changelog_merger(b'', b'this text\n', b'other text\n')
        status, lines = changelog_merger.merge_contents(merge_hook_params)
        self.assertEqual(('success', [b'other text\n', b'this text\n']), (status, list(lines)))