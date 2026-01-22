import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def PrintAsImageString(x):
    """Returns the molecules as base64 encoded PNG image or as SVG"""
    if highlightSubstructures and hasattr(x, '__sssAtoms'):
        highlightAtoms = x.__sssAtoms
    else:
        highlightAtoms = []
    useSVG = molRepresentation.lower() == 'svg'
    if InteractiveRenderer and InteractiveRenderer.isEnabled(x):
        size = [max(30, s) for s in molSize]
        return InteractiveRenderer.generateHTMLBody(x, size, useSVG=useSVG)
    elif useSVG:
        svg = Draw._moltoSVG(x, molSize, highlightAtoms, '', kekulize=True, drawOptions=drawOptions)
        svg = minidom.parseString(svg)
        svg = svg.getElementsByTagName('svg')[0]
        svg.attributes['viewbox'] = f'0 0 {molSize[0]} {molSize[1]}'
        svg.attributes['style'] = f'max-width: {molSize[0]}px; height: {molSize[1]}px;'
        svg.attributes['data-content'] = 'rdkit/molecule'
        return svg.toxml()
    else:
        data = Draw._moltoimg(x, molSize, highlightAtoms, '', returnPNG=True, kekulize=True, drawOptions=drawOptions)
        return f'<div style="width: {molSize[0]}px; height: {molSize[1]}px" data-content="rdkit/molecule"><img src="data:image/png;base64,%s" alt="Mol"/></div>' % _get_image(data)