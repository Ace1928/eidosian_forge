import os
import sys
import tempfile
from rdkit import Chem
def HideObject(self, objName):
    self.server.do('disable %s' % objName)