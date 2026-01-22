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
def _moltoSVG(mol, sz, highlights, legend, kekulize, drawOptions=None, **kwargs):
    try:
        with rdBase.BlockLogs():
            mol.GetAtomWithIdx(0).GetExplicitValence()
    except RuntimeError:
        mol.UpdatePropertyCache(False)
    kekulize = shouldKekulize(mol, kekulize)
    if not drawOptions or drawOptions.prepareMolsBeforeDrawing:
        try:
            with rdBase.BlockLogs():
                mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize)
        except ValueError:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    if drawOptions is not None:
        d2d.SetDrawOptions(drawOptions)
    d2d.drawOptions().prepareMolsBeforeDrawing = False
    bondHighlights = kwargs.get('highlightBonds', None)
    if bondHighlights is not None:
        d2d.DrawMolecule(mol, legend=legend or '', highlightAtoms=highlights or [], highlightBonds=bondHighlights)
    else:
        d2d.DrawMolecule(mol, legend=legend or '', highlightAtoms=highlights or [])
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg