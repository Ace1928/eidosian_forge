import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_diverged_with_overwrite(self):
    self.do_incremental_upload(directory=self.diverged_tree.basedir, overwrite=True)
    self.assertRevidUploaded(b'rev2b')