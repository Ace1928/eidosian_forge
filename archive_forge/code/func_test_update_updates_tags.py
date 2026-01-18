from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_update_updates_tags(self):
    master = self.make_branch('master')
    master.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
    master.tags.set_tag('tag2', b'target2')
    child.update()
    self.assertEqual(b'target2', child.tags.lookup_tag('tag2'))