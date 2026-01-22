import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def assertDuplicateEntry(self, wt, c):
    tpath = self._this['path']
    tfile_id = self._this['file_id']
    opath = self._other['path']
    ofile_id = self._other['file_id']
    self.assertEqual(tpath, opath)
    self.assertEqual(tfile_id, c.file_id)
    self.assertEqual(tpath + '.moved', c.path)
    self.assertEqual(tpath, c.conflict_path)