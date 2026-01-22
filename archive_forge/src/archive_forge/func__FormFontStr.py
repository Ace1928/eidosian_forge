from math import *
from rdkit.sping.PDF import pdfmetrics  # for font info
from rdkit.sping.pid import *
def _FormFontStr(self, font):
    """ form what we hope is a valid SVG font string.
      Defaults to 'sansserif'
      This should work when an array of font faces are passed in.
    """
    fontStr = ''
    if font.face is None:
        font.__dict__['face'] = 'sansserif'
    if isinstance(font.face, str):
        if len(font.face.split()) > 1:
            familyStr = "'%s'" % font.face
        else:
            familyStr = font.face
    else:
        face = font.face[0]
        if len(face.split()) > 1:
            familyStr = "'%s'" % face
        else:
            familyStr = face
        for i in range(1, len(font.face)):
            face = font.face[i]
            if len(face.split()) > 1:
                familyStr = ", '%s'" % face
            else:
                familyStr = familyStr + ', %s' % face
    if font.italic:
        styleStr = 'font-style="italic"'
    else:
        styleStr = ''
    if font.bold:
        weightStr = 'font-weight="bold"'
    else:
        weightStr = ''
    if font.size:
        sizeStr = 'font-size="%.2f"' % font.size
    else:
        sizeStr = ''
    fontStr = 'font-family="%s" %s %s %s' % (familyStr, styleStr, weightStr, sizeStr)
    return fontStr