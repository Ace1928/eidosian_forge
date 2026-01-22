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
def MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=None, highlightAtomLists=None, highlightBondLists=None, useSVG=False, returnPNG=False, **kwargs):
    if legends and len(legends) > len(mols):
        legends = legends[:len(mols)]
    if highlightAtomLists and len(highlightAtomLists) > len(mols):
        highlightAtomLists = highlightAtomLists[:len(mols)]
    if highlightBondLists and len(highlightBondLists) > len(mols):
        highlightBondLists = highlightBondLists[:len(mols)]
    if useSVG:
        return _MolsToGridSVG(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, legends=legends, highlightAtomLists=highlightAtomLists, highlightBondLists=highlightBondLists, **kwargs)
    else:
        return _MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, legends=legends, highlightAtomLists=highlightAtomLists, highlightBondLists=highlightBondLists, returnPNG=returnPNG, **kwargs)