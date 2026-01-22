from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
class TestFetchBase(TestCaseWithRepository):

    def make_source_branch(self):
        builder = self.make_branch_builder('source')
        content = [b'content lines\nfor the first revision\nwhich is a marginal amount of content\n']
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b''.join(content)))], revision_id=b'A-id')
        content.append(b'and some more lines for B\n')
        builder.build_snapshot([b'A-id'], [('modify', ('a', b''.join(content)))], revision_id=b'B-id')
        content.append(b'and yet even more content for C\n')
        builder.build_snapshot([b'B-id'], [('modify', ('a', b''.join(content)))], revision_id=b'C-id')
        builder.finish_series()
        source_b = builder.get_branch()
        source_b.lock_read()
        self.addCleanup(source_b.unlock)
        return (content, source_b)