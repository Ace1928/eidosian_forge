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
def MolToQPixmap(mol, size=(300, 300), kekulize=True, wedgeBonds=True, fitImage=False, options=None, **kwargs):
    """ Generates a drawing of a molecule on a Qt QPixmap
    """
    if not mol:
        raise ValueError('Null molecule provided')
    from rdkit.Chem.Draw.qtCanvas import Canvas
    canvas = Canvas(size)
    if options is None:
        options = DrawingOptions()
    options.bgColor = None
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)
    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)
    drawer.AddMol(mol, **kwargs)
    canvas.flush()
    return canvas.pixmap