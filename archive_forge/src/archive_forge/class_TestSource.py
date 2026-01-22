from breezy.bzr import vf_search
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestSource(TestCaseWithRepository):
    """Tests for/about the results of Repository._get_source."""
    scenarios = all_repository_vf_format_scenarios()

    def test_no_absent_records_in_stream_with_ghosts(self):
        builder = self.make_branch_builder('repo')
        builder.start_series()
        builder.build_snapshot([b'ghost'], [('add', ('', b'ROOT_ID', 'directory', ''))], allow_leftmost_as_ghost=True, revision_id=b'tip')
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo = b.repository
        source = repo._get_source(repo._format)
        search = vf_search.PendingAncestryResult([b'tip'], repo)
        stream = source.get_stream(search)
        for substream_type, substream in stream:
            for record in substream:
                self.assertNotEqual('absent', record.storage_kind, 'Absent record for {}'.format((substream_type,) + record.key))