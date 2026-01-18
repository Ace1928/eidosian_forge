import re
from breezy.bzr.tests.test_testament import (REV_1_SHORT, REV_1_SHORT_STRICT,
def test_testament_command_3(self):
    """Command getting short testament of previous version."""
    out, err = self.run_bzr('testament -r1 --strict')
    self.assertEqualDiff(err, '')
    self.assertEqualDiff(out, REV_1_SHORT_STRICT.decode('ascii'))