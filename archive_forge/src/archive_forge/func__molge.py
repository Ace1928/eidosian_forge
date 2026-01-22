import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def _molge(x, y):
    """Allows for substructure check using the >= operator (X has substructure Y -> X >= Y) by
    monkey-patching the __ge__ function
    This has the effect that the pandas/numpy rowfilter can be used for substructure filtering
    (filtered = dframe[dframe['RDKitColumn'] >= SubstructureMolecule])
    """
    if x is None or y is None:
        return False
    if hasattr(x, '_substructfp'):
        if not hasattr(y, '_substructfp'):
            y._substructfp = _fingerprinter(y, True)
        if not DataStructs.AllProbeBitsMatch(y._substructfp, x._substructfp):
            return False
    match = x.GetSubstructMatch(y)
    x.__sssAtoms = []
    if match:
        if highlightSubstructures:
            x.__sssAtoms = list(match)
        return True
    else:
        return False