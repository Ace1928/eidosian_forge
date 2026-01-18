from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_update_tag_into_checkout(self):
    master = self.make_branch('master')
    child = self.make_branch('child')
    child.bind(master)
    child.tags.set_tag('foo', b'rev-1')
    self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
    child.tags.delete_tag('foo')
    self.assertRaises(errors.NoSuchTag, master.tags.lookup_tag, 'foo')