import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def _check_can_encode_paths(self):
    fs_enc = sys.getfilesystemencoding()
    terminal_enc = osutils.get_terminal_encoding()
    fname = self.info['filename']
    dir_name = self.info['directory']
    for thing in [fname, dir_name]:
        try:
            thing.encode(fs_enc)
        except UnicodeEncodeError:
            raise tests.TestSkipped('Unable to represent path %r in filesystem encoding "%s"' % (thing, fs_enc))
        try:
            thing.encode(terminal_enc)
        except UnicodeEncodeError:
            raise tests.TestSkipped('Unable to represent path %r in terminal encoding "%s" (even though it is valid in filesystem encoding "%s")' % (thing, terminal_enc, fs_enc))