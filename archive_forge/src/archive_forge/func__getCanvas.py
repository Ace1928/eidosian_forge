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
def _getCanvas():
    useAGG = False
    useCairo = False
    useSping = False
    Canvas = None
    if not os.environ.get('RDKIT_CANVAS', ''):
        try:
            from rdkit.Chem.Draw.cairoCanvas import Canvas
            useCairo = True
        except ImportError:
            try:
                from rdkit.Chem.Draw.aggCanvas import Canvas
                useAGG = True
            except ImportError:
                from rdkit.Chem.Draw.spingCanvas import Canvas
                useSping = True
    else:
        canv = os.environ['RDKIT_CANVAS'].lower()
        if canv == 'cairo':
            from rdkit.Chem.Draw.cairoCanvas import Canvas
            useCairo = True
        elif canv == 'agg':
            from rdkit.Chem.Draw.aggCanvas import Canvas
            useAGG = True
        else:
            from rdkit.Chem.Draw.spingCanvas import Canvas
            useSping = True
    if useSping:
        DrawingOptions.radicalSymbol = '.'
    return (useAGG, useCairo, Canvas)