import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def _embedTeXFont(self, fontinfo):
    _log.debug('Embedding TeX font %s - fontinfo=%s', fontinfo.dvifont.texname, fontinfo.__dict__)
    widthsObject = self.reserveObject('font widths')
    self.writeObject(widthsObject, fontinfo.dvifont.widths)
    fontdictObject = self.reserveObject('font dictionary')
    fontdict = {'Type': Name('Font'), 'Subtype': Name('Type1'), 'FirstChar': 0, 'LastChar': len(fontinfo.dvifont.widths) - 1, 'Widths': widthsObject}
    if fontinfo.encodingfile is not None:
        fontdict['Encoding'] = {'Type': Name('Encoding'), 'Differences': [0, *map(Name, dviread._parse_enc(fontinfo.encodingfile))]}
    if fontinfo.fontfile is None:
        _log.warning('Because of TeX configuration (pdftex.map, see updmap option pdftexDownloadBase14) the font %s is not embedded. This is deprecated as of PDF 1.5 and it may cause the consumer application to show something that was not intended.', fontinfo.basefont)
        fontdict['BaseFont'] = Name(fontinfo.basefont)
        self.writeObject(fontdictObject, fontdict)
        return fontdictObject
    t1font = _type1font.Type1Font(fontinfo.fontfile)
    if fontinfo.effects:
        t1font = t1font.transform(fontinfo.effects)
    fontdict['BaseFont'] = Name(t1font.prop['FontName'])
    effects = (fontinfo.effects.get('slant', 0.0), fontinfo.effects.get('extend', 1.0))
    fontdesc = self.type1Descriptors.get((fontinfo.fontfile, effects))
    if fontdesc is None:
        fontdesc = self.createType1Descriptor(t1font, fontinfo.fontfile)
        self.type1Descriptors[fontinfo.fontfile, effects] = fontdesc
    fontdict['FontDescriptor'] = fontdesc
    self.writeObject(fontdictObject, fontdict)
    return fontdictObject