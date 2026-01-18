import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_take_this(self):
    opts, args = self.parse(['--action', 'take-this'])
    self.assertEqual({'action': 'take_this'}, opts)
    opts, args = self.parse(['--take-this'])
    self.assertEqual({'action': 'take_this'}, opts)