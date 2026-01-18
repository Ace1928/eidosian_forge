from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def test_sprout_from_stacked_with_short_history(self):
    content, source_b = self.make_source_branch()
    stack_b = self.make_branch('stack-on')
    stack_b.pull(source_b, stop_revision=b'B-id')
    target_b = self.make_branch('target')
    target_b.set_stacked_on_url('../stack-on')
    target_b.pull(source_b, stop_revision=b'C-id')
    final_b = self.make_branch('final')
    final_b.pull(target_b)
    final_b.lock_read()
    self.addCleanup(final_b.unlock)
    self.assertEqual(b'C-id', final_b.last_revision())
    text_keys = [(b'a-id', b'A-id'), (b'a-id', b'B-id'), (b'a-id', b'C-id')]
    stream = final_b.repository.texts.get_record_stream(text_keys, 'unordered', True)
    records = sorted([(r.key, r.get_bytes_as('fulltext')) for r in stream])
    self.assertEqual([((b'a-id', b'A-id'), b''.join(content[:-2])), ((b'a-id', b'B-id'), b''.join(content[:-1])), ((b'a-id', b'C-id'), b''.join(content))], records)