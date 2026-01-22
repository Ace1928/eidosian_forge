import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def _updateFillColor(self, color):
    """Triggered when someone assigns to defaultFillColor"""
    self.pdf.setFillColorRGB(color.red, color.green, color.blue)