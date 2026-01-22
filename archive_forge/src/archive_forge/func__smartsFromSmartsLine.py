import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def _smartsFromSmartsLine(line):
    """
    Converts given line into a molecule using 'Chem.MolFromSmarts'.
    """
    whitespace = re.compile('[\\t ]+')
    line = line.strip().split('//')[0]
    if line:
        smarts = whitespace.split(line)
        salt = Chem.MolFromSmarts(smarts[0])
        if salt is None:
            raise ValueError(line)
        return salt