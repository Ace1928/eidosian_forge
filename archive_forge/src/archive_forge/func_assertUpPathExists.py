import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def assertUpPathExists(self, path, base=upload_dir):
    self.assertPathExists(osutils.pathjoin(base, path))