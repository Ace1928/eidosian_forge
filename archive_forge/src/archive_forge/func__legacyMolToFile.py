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
def _legacyMolToFile(mol, fileName, size, kekulize, wedgeBonds, imageType, fitImage, options, **kwargs):
    """ Generates a drawing of a molecule and writes it to a file
  """
    if options is None:
        options = DrawingOptions()
    useAGG, useCairo, Canvas = _getCanvas()
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    if useCairo or useAGG:
        canvas = Canvas(size=size, imageType=imageType, fileName=fileName)
    else:
        options.radicalSymbol = '.'
        canvas = Canvas(size=size, name=fileName, imageType=imageType)
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)
    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)
    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
    drawer.AddMol(mol, **kwargs)
    if useCairo or useAGG:
        canvas.flush()
    else:
        canvas.save()