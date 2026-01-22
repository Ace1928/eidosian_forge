import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def assertParentLoop(self, wt, c):
    self.assertEqual(self._other['dir_id'], c.file_id)
    self.assertEqual(self._other['target_id'], c.conflict_file_id)
    if self._other['xfail']:
        self.knownFailure("ParentLoop doesn't carry enough info to resolve --take-other")