import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_run_editor(self):
    self.overrideEnv('BRZ_EDITOR', self.make_do_nothing_editor())
    self.assertEqual(True, msgeditor._run_editor(''), 'Unable to run dummy fake editor')