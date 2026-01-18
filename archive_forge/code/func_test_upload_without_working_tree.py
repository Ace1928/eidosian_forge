import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_without_working_tree(self):
    self.do_full_upload(directory=self.remote_branch_url)
    self.assertUpFileEqual(b'foo', 'hello')