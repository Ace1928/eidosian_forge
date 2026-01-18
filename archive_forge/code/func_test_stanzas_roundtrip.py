import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_stanzas_roundtrip(self):
    stanzas_iter = bzr_conflicts.ConflictList(example_conflicts).to_stanzas()
    processed = bzr_conflicts.ConflictList.from_stanzas(stanzas_iter)
    self.assertEqual(example_conflicts, processed)