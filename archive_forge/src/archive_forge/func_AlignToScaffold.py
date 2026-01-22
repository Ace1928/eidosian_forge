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
def AlignToScaffold(frame, molCol='ROMol', scaffoldCol='Murcko_SMILES'):
    """
    Aligns molecules in molCol to scaffolds in scaffoldCol
    """
    frame[molCol] = frame.apply(lambda x: AlignMol(x[molCol], x[scaffoldCol]), axis=1)