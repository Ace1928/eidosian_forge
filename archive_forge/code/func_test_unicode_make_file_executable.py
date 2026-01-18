import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_unicode_make_file_executable(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    self._test_make_file_executable('helloØ')