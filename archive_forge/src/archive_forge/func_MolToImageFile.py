import os
import warnings
from collections import namedtuple
from importlib.util import find_spec
from io import BytesIO
import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw.rdMolDraw2D import *
def MolToImageFile(mol, filename, size=(300, 300), kekulize=True, wedgeBonds=True, **kwargs):
    """  DEPRECATED:  please use MolToFile instead

  """
    warnings.warn('MolToImageFile is deprecated, please use MolToFile instead', DeprecationWarning)
    img = MolToImage(mol, size=size, kekulize=kekulize, wedgeBonds=wedgeBonds, **kwargs)
    img.save(filename)