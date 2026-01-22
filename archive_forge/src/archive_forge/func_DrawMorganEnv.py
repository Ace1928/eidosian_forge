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
def DrawMorganEnv(mol, atomId, radius, molSize=(150, 150), baseRad=0.3, useSVG=True, aromaticColor=(0.9, 0.9, 0.2), ringColor=(0.8, 0.8, 0.8), centerColor=(0.6, 0.6, 0.9), extraColor=(0.9, 0.9, 0.9), drawOptions=None, **kwargs):
    menv = _getMorganEnv(mol, atomId, radius, baseRad, aromaticColor, ringColor, centerColor, extraColor, **kwargs)
    if useSVG:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    if drawOptions is None:
        drawOptions = drawer.drawOptions()
    drawOptions.continuousHighlight = False
    drawOptions.includeMetadata = False
    drawer.SetDrawOptions(drawOptions)
    drawer.DrawMolecule(menv.submol, highlightAtoms=menv.highlightAtoms, highlightAtomColors=menv.atomColors, highlightBonds=menv.highlightBonds, highlightBondColors=menv.bondColors, highlightAtomRadii=menv.highlightRadii, **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()