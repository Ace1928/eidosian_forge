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
def _update_working_trees(self, trees):
    if self.verbose:
        reporter = delta._ChangeReporter()
    else:
        reporter = None
    for wt in trees:
        self.note('Updating the working tree for %s ...', wt.basedir)
        wt.update(reporter)
        self._tree_count += 1