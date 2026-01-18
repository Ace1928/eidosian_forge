import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_take_other(self):
    opts, args = self.parse(['--action', 'take-other'])
    self.assertEqual({'action': 'take_other'}, opts)
    opts, args = self.parse(['--take-other'])
    self.assertEqual({'action': 'take_other'}, opts)