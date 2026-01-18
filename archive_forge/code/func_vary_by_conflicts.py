import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def vary_by_conflicts():
    for conflict in example_conflicts:
        yield (conflict.__class__.__name__, {'conflict': conflict})