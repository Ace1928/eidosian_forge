import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_set_commit_message_hook(self):
    msgeditor.hooks.install_named_hook('set_commit_message', lambda commit_obj, existing_message: 'save me some typing\n', None)
    commit_obj = commit.Commit()
    self.assertEqual('save me some typing\n', msgeditor.set_commit_message(commit_obj))