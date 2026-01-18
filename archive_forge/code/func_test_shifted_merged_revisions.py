import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_shifted_merged_revisions(self):
    """Test irregular layout.

        Requesting revisions touching a file can produce "holes" in the depths.
        """
    self.assertReversed([('1', 0), ('2', 0), ('1.1', 2), ('1.2', 2)], [('2', 0), ('1.2', 2), ('1.1', 2), ('1', 0)])