import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_resolve_taking_this(self):
    self.run_script("\n$ brz resolve --take-this foo.new\n2>...\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")