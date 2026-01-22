import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
class PsDSC:

    def __init__(self):
        pass

    def documentHeader(self):
        return '%!PS-Adobe-3.0'

    def boundingBoxStr(self, x0, y0, x1, y1):
        """coordinates of bbox in default PS coordinates"""
        return '%%BoundingBox: ' + '%s %s %s %s' % (x0, y0, x1, y1)

    def BeginPageStr(self, pageSetupStr, pageName=None):
        """Use this at the beginning of each page, feed it your setup code
        in the form of a string of postscript.  pageName is the "number" of the
        page.  By default it will be 0."""
        self.inPageFlag = 1
        if not pageName:
            pageDeclaration = '%%Page: %d %d' % (1, 1)
        else:
            pageDeclaration = '%%Page: ' + pageName
        ret = pageDeclaration + '\n' + '%%BeginPageSetup\n/pgsave save def\n'
        return ret + pageSetupStr + '\n%%EndPageSetup'

    def EndPageStr(self):
        self.inPageFlag = 0
        return ''