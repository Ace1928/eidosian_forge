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
def _toHTML(mol):
    useInteractiveRenderer = InteractiveRenderer.isEnabled(mol)
    if _canUse3D and ipython_3d and mol.GetNumConformers() and mol.GetConformer().Is3D():
        return _toJSON(mol)
    props = mol.GetPropsAsDict()
    if not ipython_showProperties or not props:
        if useInteractiveRenderer:
            return _wrapHTMLIntoTable(InteractiveRenderer.generateHTMLBody(mol, molSize, useSVG=ipython_useSVG))
        else:
            return _toSVG(mol)
    if mol.HasProp('_Name'):
        nm = mol.GetProp('_Name')
    else:
        nm = ''
    res = []
    if useInteractiveRenderer:
        content = InteractiveRenderer.generateHTMLBody(mol, molSize, legend=nm, useSVG=ipython_useSVG)
    elif not ipython_useSVG:
        png = Draw._moltoimg(mol, molSize, [], nm, returnPNG=True, kekulize=kekulizeStructures, drawOptions=drawOptions)
        png = base64.b64encode(png)
        content = f'<image src="data:image/png;base64,{png.decode()}">'
    else:
        content = Draw._moltoSVG(mol, molSize, [], nm, kekulize=kekulizeStructures, drawOptions=drawOptions)
    res.append(f'<tr><td colspan="2" style="text-align: center;">{content}</td></tr>')
    for i, (pn, pv) in enumerate(props.items()):
        if ipython_maxProperties >= 0 and i >= ipython_maxProperties:
            res.append('<tr><td colspan="2" style="text-align: center">Property list truncated.<br />Increase IPythonConsole.ipython_maxProperties (or set it to -1) to see more properties.</td></tr>')
            break
        pv = html.escape(str(pv))
        res.append(f'<tr><th style="text-align: right">{pn}</th><td style="text-align: left">{pv}</td></tr>')
    res = '\n'.join(res)
    res = f'<table>{res}</table>'
    if useInteractiveRenderer:
        res = InteractiveRenderer.injectHTMLFooterAfterTable(res)
    return res