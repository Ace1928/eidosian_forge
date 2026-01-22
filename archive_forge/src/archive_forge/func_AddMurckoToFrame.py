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
def AddMurckoToFrame(frame, molCol='ROMol', MurckoCol='Murcko_SMILES', Generic=False):
    """
    Adds column with SMILES of Murcko scaffolds to pandas DataFrame.

    Generic set to true results in SMILES of generic framework.
    """
    if Generic:

        def func(x):
            return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(x[molCol])))
    else:

        def func(x):
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(x[molCol]))
    frame[MurckoCol] = frame.apply(func, axis=1)