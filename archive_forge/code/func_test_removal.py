import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_removal(self):
    self.assertReport(' D  path/', modified='deleted', kind=('directory', None), old_path='old')
    self.assertReport('-   path/', versioned_change='removed', old_path='path', kind=(None, 'directory'))
    self.assertReport('-D  path', versioned_change='removed', old_path='path', modified='deleted', kind=('file', 'directory'))