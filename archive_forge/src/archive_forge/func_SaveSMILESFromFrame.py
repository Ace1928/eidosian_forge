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
def SaveSMILESFromFrame(frame, outFile, molCol='ROMol', NamesCol='', isomericSmiles=False):
    """
    Saves smi file. SMILES are generated from column with RDKit molecules. Column
    with names is optional.
    """
    w = Chem.SmilesWriter(outFile, isomericSmiles=isomericSmiles)
    if NamesCol != '':
        for m, n in zip(frame[molCol], (str(c) for c in frame[NamesCol])):
            m.SetProp('_Name', n)
            w.write(m)
        w.close()
    else:
        for m in frame[molCol]:
            w.write(m)
        w.close()