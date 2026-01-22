import os
import sys
import tempfile
from rdkit import Chem
def DisplayHBonds(self, objName, molName, proteinName, molSelText='(%(molName)s)', proteinSelText='(%(proteinName)s and not het)'):
    """ toggles display of h bonds between the protein and a specified molecule """
    cmd = 'delete %(objName)s;\n'
    cmd += 'dist %(objName)s,' + molSelText + ',' + proteinSelText + ',mode=2;\n'
    cmd += 'enable %(objName)s;'
    cmd = cmd % locals()
    self.server.do(cmd)