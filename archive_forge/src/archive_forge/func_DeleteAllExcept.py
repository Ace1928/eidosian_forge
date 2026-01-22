import os
import sys
import tempfile
from rdkit import Chem
def DeleteAllExcept(self, excludes):
    """ deletes everything except the items in the provided list of arguments """
    allNames = self.server.getNames('*', False)
    for nm in allNames:
        if nm not in excludes:
            self.server.deleteObject(nm)