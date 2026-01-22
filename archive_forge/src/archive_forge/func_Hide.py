import os
import tempfile
from win32com.client import Dispatch
from rdkit import Chem
def Hide(self, recurse=True):
    self.Select(state=True, recurse=True)
    self.doc.DoCommand('hide')
    self.Select(state=False, recurse=True)