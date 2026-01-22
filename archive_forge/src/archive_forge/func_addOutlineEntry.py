import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def addOutlineEntry(self, title, key, level=0, closed=None):
    """Adds a new entry to the outline at given level.  If LEVEL not specified,
        entry goes at the top level.  If level specified, it must be
        no more than 1 greater than the outline level in the last call.

        The key must be the (unique) name of a bookmark.
        the title is the (non-unique) name to be displayed for the entry.

        If closed is set then the entry should show no subsections by default
        when displayed.

        Example::
        
           c.addOutlineEntry("first section", "section1")
           c.addOutlineEntry("introduction", "s1s1", 1, closed=1)
           c.addOutlineEntry("body", "s1s2", 1)
           c.addOutlineEntry("detail1", "s1s2s1", 2)
           c.addOutlineEntry("detail2", "s1s2s2", 2)
           c.addOutlineEntry("conclusion", "s1s3", 1)
           c.addOutlineEntry("further reading", "s1s3s1", 2)
           c.addOutlineEntry("second section", "section1")
           c.addOutlineEntry("introduction", "s2s1", 1)
           c.addOutlineEntry("body", "s2s2", 1, closed=1)
           c.addOutlineEntry("detail1", "s2s2s1", 2)
           c.addOutlineEntry("detail2", "s2s2s2", 2)
           c.addOutlineEntry("conclusion", "s2s3", 1)
           c.addOutlineEntry("further reading", "s2s3s1", 2)

        generated outline looks like::
        
            - first section
            |- introduction
            |- body
            |  |- detail1
            |  |- detail2
            |- conclusion
            |  |- further reading
            - second section
            |- introduction
            |+ body
            |- conclusion
            |  |- further reading

        Note that the second "body" is closed.

        Note that you can jump from level 5 to level 3 but not
        from 3 to 5: instead you need to provide all intervening
        levels going down (4 in this case).  Note that titles can
        collide but keys cannot.
        """
    self._doc.outline.addOutlineEntry(key, level, title, closed=closed)