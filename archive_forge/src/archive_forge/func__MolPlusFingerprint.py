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
def _MolPlusFingerprint(m):
    """Precomputes fingerprints and stores results in molecule objects to accelerate
       substructure matching
    """
    if m is not None:
        m._substructfp = _fingerprinter(m, False)
    return m