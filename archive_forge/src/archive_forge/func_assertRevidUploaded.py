import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def assertRevidUploaded(self, revid):
    t = self.get_transport(self.upload_dir)
    uploaded_revid = t.get_bytes('.bzr-upload.revid')
    self.assertEqual(revid, uploaded_revid)