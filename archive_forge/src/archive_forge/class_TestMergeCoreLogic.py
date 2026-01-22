from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
class TestMergeCoreLogic(tests.TestCase):

    def test_new_in_other_floats_to_top(self):
        """Changes at the top of 'other' float to the top.

        Given a changelog in THIS containing::

          NEW-1
          OLD-1

        and a changelog in OTHER containing::

          NEW-2
          OLD-1

        it will merge as::

          NEW-2
          NEW-1
          OLD-1
        """
        base_entries = [b'OLD-1']
        this_entries = [b'NEW-1', b'OLD-1']
        other_entries = [b'NEW-2', b'OLD-1']
        result_entries = changelog_merge.merge_entries(base_entries, this_entries, other_entries)
        self.assertEqual([b'NEW-2', b'NEW-1', b'OLD-1'], result_entries)

    def test_acceptance_bug_723968(self):
        """Merging a branch that:

         1. adds a new entry, and
         2. edits an old entry (e.g. to fix a typo or twiddle formatting)

        will:

         1. add the new entry to the top
         2. keep the edit, without duplicating the edited entry or moving it.
        """
        result_entries = changelog_merge.merge_entries(sample_base_entries, sample_this_entries, sample_other_entries)
        self.assertEqual([b'Other entry O1', b'This entry T1', b'This entry T2', b'Base entry B1', b'Base entry B2 updated', b'Base entry B3'], list(result_entries))

    def test_more_complex_conflict(self):
        """Like test_acceptance_bug_723968, but with a more difficult conflict:
        the new entry and the edited entry are adjacent.
        """

        def guess_edits(new, deleted):
            return changelog_merge.default_guess_edits(new, deleted, entry_as_str=lambda x: x)
        result_entries = changelog_merge.merge_entries(sample2_base_entries, sample2_this_entries, sample2_other_entries, guess_edits=guess_edits)
        self.assertEqual([b'Other entry O1', b'This entry T1', b'This entry T2', b'Base entry B1 edit', b'Base entry B2'], list(result_entries))

    def test_too_hard(self):
        """A conflict this plugin cannot resolve raises EntryConflict.
        """
        self.assertRaises(changelog_merge.EntryConflict, changelog_merge.merge_entries, [(entry,) for entry in sample2_base_entries], [], [(entry,) for entry in sample2_other_entries])

    def test_default_guess_edits(self):
        """default_guess_edits matches a new entry only once.

        (Even when that entry is the best match for multiple old entries.)
        """
        new_in_other = [(b'AAAAA',), (b'BBBBB',)]
        deleted_in_other = [(b'DDDDD',), (b'BBBBBx',), (b'BBBBBxx',)]
        result = changelog_merge.default_guess_edits(new_in_other, deleted_in_other)
        self.assertEqual(([(b'AAAAA',)], [(b'DDDDD',), (b'BBBBBxx',)], [((b'BBBBBx',), (b'BBBBB',))]), result)