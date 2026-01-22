import time
import configobj
from fastimport import commands
from fastimport import errors as plugin_errors
from fastimport import processor
from fastimport.helpers import invert_dictset
from .... import debug, delta, errors, osutils, progress
from .... import revision as _mod_revision
from ....bzr.knitpack_repo import KnitPackRepository
from ....trace import mutter, note, warning
from .. import (branch_updater, cache_manager, helpers, idmapfile, marks_file,
def _set_tag(self, name, from_):
    """Define a tag given a name and import 'from' reference."""
    bzr_tag_name = name.decode('utf-8', 'replace')
    bzr_rev_id = self.cache_mgr.lookup_committish(from_)
    self.tags[bzr_tag_name] = bzr_rev_id