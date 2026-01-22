import os
import sys
import tempfile
from rdkit import Chem
def SaveFile(self, filename):
    if not filename:
        raise ValueError('empty filename')
    filename = os.path.abspath(filename)
    self.server.save(filename)