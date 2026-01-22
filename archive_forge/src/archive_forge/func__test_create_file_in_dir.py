import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _test_create_file_in_dir(self, dir_name, file_name):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    self.add_dir(dir_name)
    fpath = '{}/{}'.format(dir_name, file_name)
    self.add_file(fpath, b'baz')
    self.assertUpPathDoesNotExist(fpath)
    self.do_upload()
    self.assertUpFileEqual(b'baz', fpath)
    self.assertUpPathModeEqual(dir_name, 509)