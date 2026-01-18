from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def test_sprout_from_smart_stacked_with_short_history(self):
    content, source_b = self.make_source_branch()
    transport = self.make_smart_server('server')
    transport.ensure_base()
    url = transport.abspath('')
    stack_b = source_b.controldir.sprout(url + '/stack-on', revision_id=b'B-id')
    target_transport = transport.clone('target')
    target_transport.ensure_base()
    target_bzrdir = self.bzrdir_format.initialize_on_transport(target_transport)
    target_bzrdir.create_repository()
    target_b = target_bzrdir.create_branch()
    target_b.set_stacked_on_url('../stack-on')
    target_b.pull(source_b, stop_revision=b'C-id')
    final_b = target_b.controldir.sprout('final').open_branch()
    self.assertEqual(b'C-id', final_b.last_revision())
    final2_b = target_b.controldir.sprout('final2', revision_id=b'C-id').open_branch()
    self.assertEqual(b'C-id', final_b.last_revision())