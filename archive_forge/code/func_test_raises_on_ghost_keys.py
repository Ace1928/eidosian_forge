from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def test_raises_on_ghost_keys(self):
    vf = self.make_vf()
    gen = versionedfile._MPDiffGenerator(vf, [(b'one',)])
    self.assertRaises(errors.RevisionNotPresent, gen._find_needed_keys)