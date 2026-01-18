import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_simple_symbolic_link_loop(self):
    filepaths = ['Left.toRight.toLeft', 'Left.toRight', 'Left', 'Level0.Level1.Level2.Level3', 'Level0.Level1.Level2', 'Level0.Level1', 'Level0', 'Right', 'Right.toLeft', 'Right.toLeft.toRight']
    for thing in self.module_gatherer.modules:
        self.assertIn(thing, filepaths)
        if thing == 'Left.toRight.toLeft':
            filepaths.remove('Right.toLeft')
            filepaths.remove('Right.toLeft.toRight')
        if thing == 'Right.toLeft.toRight':
            filepaths.remove('Left.toRight.toLeft')
            filepaths.remove('Left.toRight')
        filepaths.remove(thing)
    self.assertFalse(filepaths)