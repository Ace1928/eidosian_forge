import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_log_with_tags(self):
    wt = self._prepare_tree_with_merges(with_tags=True)
    self.assertFormatterResult(b'    3 Joe Foo\t2005-11-22 {v1.0, v1.0rc1}\n      rev-3\n\n    2 Joe Foo\t2005-11-22 {v0.2} [merge]\n      rev-2\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\n', wt.branch, log.ShortLogFormatter)