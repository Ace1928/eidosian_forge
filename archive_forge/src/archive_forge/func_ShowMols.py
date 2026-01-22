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
def ShowMols(mols, maxMols=50, **kwargs):
    global _MolsToGridImageSaved
    if 'useSVG' not in kwargs:
        kwargs['useSVG'] = ipython_useSVG
    if 'returnPNG' not in kwargs:
        kwargs['returnPNG'] = True
    if InteractiveRenderer.isEnabled():
        fn = InteractiveRenderer.MolsToHTMLTable
    elif _MolsToGridImageSaved is not None:
        fn = _MolsToGridImageSaved
    else:
        fn = Draw.MolsToGridImage
    if len(mols) > maxMols:
        warnings.warn('Truncating the list of molecules to be displayed to %d. Change the maxMols value to display more.' % maxMols)
        mols = mols[:maxMols]
        for prop in ('legends', 'highlightAtoms', 'highlightBonds'):
            if prop in kwargs:
                kwargs[prop] = kwargs[prop][:maxMols]
    if 'drawOptions' not in kwargs:
        kwargs['drawOptions'] = drawOptions
    res = fn(mols, **kwargs)
    if InteractiveRenderer.isEnabled():
        return HTML(res)
    elif kwargs['useSVG']:
        return SVG(res)
    elif kwargs['returnPNG']:
        return display.Image(data=res, format='png')
    return res