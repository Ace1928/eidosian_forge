import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_start_message(self):
    self.make_uncommitted_tree()
    self.make_fake_editor()
    self.assertEqual('test message from fed\nstart message\n', msgeditor.edit_commit_message('', start_message='start message\n'))
    self.assertEqual('test message from fed\n', msgeditor.edit_commit_message('', start_message=''))