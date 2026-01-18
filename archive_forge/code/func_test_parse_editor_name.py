import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_parse_editor_name(self):
    """Correctly interpret names with spaces.

        See <https://bugs.launchpad.net/bzr/+bug/220331>
        """
    self.overrideEnv('BRZ_EDITOR', '"%s"' % self.make_do_nothing_editor('name with spaces'))
    self.assertEqual(True, msgeditor._run_editor('a_filename'))