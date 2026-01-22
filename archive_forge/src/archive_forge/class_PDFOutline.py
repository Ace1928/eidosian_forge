import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
class PDFOutline(PDFObject):
    """null outline, does nothing yet"""

    def __init__(self):
        self.template = LINEEND.join(['<<', '/Type /Outlines', '/Count 0', '>>'])

    def save(self, file):
        file.write(self.template + LINEEND)