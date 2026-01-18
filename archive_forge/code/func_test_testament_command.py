import re
from breezy.bzr.tests.test_testament import (REV_1_SHORT, REV_1_SHORT_STRICT,
def test_testament_command(self):
    """Testament containing a file and a directory."""
    out, err = self.run_bzr('testament --long')
    self.assertEqualDiff(err, '')
    self.assertEqualDiff(out, REV_2_TESTAMENT.decode('ascii'))