import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def _resetDefaults(self):
    """Only used in setup - persist from page to page"""
    self.defaultLineColor = black
    self.defaultFillColor = transparent
    self.defaultLineWidth = 1
    self.defaultFont = Font()
    self.pdf.setLineCap(2)