import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def _updateFont(self, font):
    """Triggered when someone assigns to defaultFont"""
    psfont = self._findPostScriptFontName(font)
    self.pdf.setFont(psfont, font.size)