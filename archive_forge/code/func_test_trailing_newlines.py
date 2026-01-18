import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_trailing_newlines(self):
    wt = self.make_branch_and_tree('.')
    b = self.make_commits_with_trailing_newlines(wt)
    self.assertFormatterResult(b'3: Joe Foo 2005-11-22 single line with trailing newline\n2: Joe Foo 2005-11-22 multiline\n1: Joe Foo 2005-11-22 simple log message\n', b, log.LineLogFormatter)