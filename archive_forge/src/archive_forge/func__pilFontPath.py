from rdkit.sping.pid import *
import math
import os
def _pilFontPath(face, size, bold=0):
    if face == 'monospaced':
        face = 'courier'
    elif face == 'serif':
        face = 'times'
    elif face == 'sansserif' or face == 'system':
        face = 'helvetica'
    if bold and face != 'symbol':
        fname = '%s-bold-%d.pil' % (face, size)
    else:
        fname = '%s-%d.pil' % (face, size)
    path = os.path.join(_fontprefix, fname)
    return path