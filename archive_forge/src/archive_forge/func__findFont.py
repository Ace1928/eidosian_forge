import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def _findFont(self, font):
    requested = font.face or 'Serif'
    if isinstance(requested, str):
        requested = [requested]
    face = PiddleLegalFonts['serif'].lower()
    for reqFace in requested:
        if reqFace.lower() in PiddleLegalFonts:
            face = PiddleLegalFonts[reqFace.lower()].lower()
            break
    if font.bold:
        shape = Bold
    elif font.italic:
        shape = Italic
    else:
        shape = Roman
    return self.fontMapEncoding[face, shape]