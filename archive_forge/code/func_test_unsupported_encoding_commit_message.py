import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_unsupported_encoding_commit_message(self):
    self.overrideEnv('LANG', 'C')
    char = probe_bad_non_ascii(osutils.get_user_encoding())
    if char is None:
        self.skipTest('Cannot find suitable non-ascii character for user_encoding (%s)' % osutils.get_user_encoding())
    self.make_fake_editor(message=char)
    self.make_uncommitted_tree()
    self.assertRaises(msgeditor.BadCommitMessageEncoding, msgeditor.edit_commit_message, '')