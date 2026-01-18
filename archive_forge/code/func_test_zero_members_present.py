import errno
from .. import osutils, tests
from . import features
def test_zero_members_present(self):
    self.build_tree(['foo'])
    st = self.win32_lstat('foo')
    self.assertEqual(0, st.st_dev)
    self.assertEqual(0, st.st_ino)
    self.assertEqual(0, st.st_uid)
    self.assertEqual(0, st.st_gid)