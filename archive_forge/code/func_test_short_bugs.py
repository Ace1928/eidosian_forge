import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_bugs(self):
    tree = self.make_commits_with_bugs()
    self.assertFormatterResult(b'    2 Joe Bar\t2005-11-22\n      fixes bugs: test://bug/id test://bug/2\n      multiline\n      log\n      message\n\n    1 Joe Foo\t2005-11-22\n      fixes bug: test://bug/id\n      simple log message\n\n', tree.branch, log.ShortLogFormatter)