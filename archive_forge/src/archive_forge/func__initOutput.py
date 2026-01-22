from math import *
from rdkit.sping.PDF import pdfmetrics  # for font info
from rdkit.sping.pid import *
def _initOutput(self, includeXMLHeader=True, extraHeaderText=''):
    if includeXMLHeader:
        self._txt = SVG_HEADER
    else:
        self._txt = ''
    self._txt += '<svg:svg version="1.1" baseProfile="full"\n        xmlns:svg="http://www.w3.org/2000/svg"\n        xmlns:xlink="http://www.w3.org/1999/xlink"\n        xml:space="preserve" width="%dpx" height="%dpx" %s>\n' % (self.size[0], self.size[1], extraHeaderText)