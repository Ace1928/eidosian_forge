from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_omitted_revision(self):
    md = self.get_MD(['-r-2..'])
    self.assertEqual(b'rev2', md.base_revision_id)
    self.assertEqual(b'rev3', md.revision_id)
    md = self.get_MD(['-r..3', '--from', 'branch', 'grandparent'], wd='.')
    self.assertEqual(b'rev1', md.base_revision_id)
    self.assertEqual(b'rev3', md.revision_id)