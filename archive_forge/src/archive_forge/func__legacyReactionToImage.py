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
def _legacyReactionToImage(rxn, subImgSize=(200, 200), **kwargs):
    from PIL import Image
    mols = []
    for i in range(rxn.GetNumReactantTemplates()):
        tmpl = rxn.GetReactantTemplate(i)
        tmpl.UpdatePropertyCache(False)
        mols.append(tmpl)
    mols.append(None)
    for i in range(rxn.GetNumProductTemplates()):
        tmpl = rxn.GetProductTemplate(i)
        tmpl.UpdatePropertyCache(False)
        mols.append(tmpl)
    res = Image.new('RGBA', (subImgSize[0] * len(mols), subImgSize[1]), (255, 255, 255, 0))
    for i, mol in enumerate(mols):
        if mol is not None:
            nimg = MolToImage(mol, subImgSize, kekulize=False, **kwargs)
        else:
            nimg, canvas = _createCanvas(subImgSize)
            p0 = (10, subImgSize[1] // 2)
            p1 = (subImgSize[0] - 10, subImgSize[1] // 2)
            p3 = (subImgSize[0] - 20, subImgSize[1] // 2 - 10)
            p4 = (subImgSize[0] - 20, subImgSize[1] // 2 + 10)
            canvas.addCanvasLine(p0, p1, lineWidth=2, color=(0, 0, 0))
            canvas.addCanvasLine(p3, p1, lineWidth=2, color=(0, 0, 0))
            canvas.addCanvasLine(p4, p1, lineWidth=2, color=(0, 0, 0))
            if hasattr(canvas, 'flush'):
                canvas.flush()
            else:
                canvas.save()
        res.paste(nimg, (i * subImgSize[0], 0))
    return res