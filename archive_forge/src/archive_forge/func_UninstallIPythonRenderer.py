import base64
import copy
import html
import warnings
from io import BytesIO
import IPython
from IPython.display import HTML, SVG
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from . import InteractiveRenderer
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from IPython import display
def UninstallIPythonRenderer():
    global _MolsToGridImageSaved, _DrawRDKitBitSaved, _DrawMorganBitSaved, _DrawMorganBitsSaved
    global _rendererInstalled, _methodsToDelete
    if not _rendererInstalled:
        return
    for cls, attr in _methodsToDelete:
        delattr(cls, attr)
    _methodsToDelete = []
    DisableSubstructMatchRendering()
    if _MolsToGridImageSaved is not None:
        Draw.MolsToGridImage = _MolsToGridImageSaved
    if _DrawRDKitBitSaved is not None:
        Draw.DrawRDKitBit = _DrawRDKitBitSaved
    if _DrawRDKitBitsSaved is not None:
        Draw.DrawRDKitBits = _DrawRDKitBitsSaved
    if _DrawMorganBitSaved is not None:
        Draw.DrawMorganBit = _DrawMorganBitSaved
    if _DrawMorganBitsSaved is not None:
        Draw.DrawMorganBits = _DrawMorganBitsSaved
    if hasattr(rdchem.Mol, '__DebugMol'):
        rdchem.Mol.Debug = rdchem.Mol.__DebugMol
        del rdchem.Mol.__DebugMol
    _rendererInstalled = False