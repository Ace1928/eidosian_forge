import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
def _expandWidths(self, compactWidthArray):
    """Expands Adobe nested list structure to get a dictionary of widths.

        Here is an example of such a structure.::
        
            (
            # starting at character ID 1, next n  characters have the widths given.
            1,  (277,305,500,668,668,906,727,305,445,445,508,668,305,379,305,539),
            # all Characters from ID 17 to 26 are 668 em units wide
            17, 26, 668,
            27, (305, 305, 668, 668, 668, 566, 871, 727, 637, 652, 699, 574, 555,
                 676, 687, 242, 492, 664, 582, 789, 707, 734, 582, 734, 605, 605,
                 641, 668, 727, 945, 609, 609, 574, 445, 668, 445, 668, 668, 590,
                 555, 609, 547, 602, 574, 391, 609, 582, 234, 277, 539, 234, 895,
                 582, 605, 602, 602, 387, 508, 441, 582, 562, 781, 531, 570, 555,
                 449, 246, 449, 668),
            # these must be half width katakana and the like.
            231, 632, 500
            )
        
        """
    data = compactWidthArray[:]
    widths = {}
    while data:
        start, data = (data[0], data[1:])
        if isSeq(data[0]):
            items, data = (data[0], data[1:])
            for offset in range(len(items)):
                widths[start + offset] = items[offset]
        else:
            end, width, data = (data[0], data[1], data[2:])
            for idx in range(start, end + 1):
                widths[idx] = width
    return widths