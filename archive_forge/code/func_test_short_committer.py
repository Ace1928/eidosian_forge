import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_committer(self):
    self.assertFormatterResult(log.ShortLogFormatter, 'committer', b'    1 Lorem Ipsum\t2005-11-22\n      add a\n\n')