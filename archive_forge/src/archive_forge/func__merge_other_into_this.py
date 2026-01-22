import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def _merge_other_into_this(self):
    b = self.builder.get_branch()
    wt = b.controldir.sprout('branch').open_workingtree()
    wt.merge_from_branch(b, b'other')
    return wt