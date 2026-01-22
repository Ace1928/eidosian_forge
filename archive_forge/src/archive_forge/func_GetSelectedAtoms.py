import os
import sys
import tempfile
from rdkit import Chem
def GetSelectedAtoms(self, whichSelection=None):
    """ returns the selected atoms """
    if not whichSelection:
        sels = self.server.getNames('selections')
        if sels:
            whichSelection = sels[-1]
        else:
            whichSelection = None
    if whichSelection:
        items = self.server.index(whichSelection)
    else:
        items = []
    return items