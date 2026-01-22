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
def FrameToGridImage(frame, column='ROMol', legendsCol=None, **kwargs):
    """
    Draw grid image of mols in pandas DataFrame.
    """
    if legendsCol:
        if legendsCol == frame.index.name:
            kwargs['legends'] = [str(c) for c in frame.index]
        else:
            kwargs['legends'] = [str(c) for c in frame[legendsCol]]
    return Draw.MolsToGridImage(list(frame[column]), **kwargs)