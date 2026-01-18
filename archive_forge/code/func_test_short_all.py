import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_all(self):
    self.assertFormatterResult(log.ShortLogFormatter, 'all', b'    1 John Doe, Jane Rey\t2005-11-22\n      add a\n\n')