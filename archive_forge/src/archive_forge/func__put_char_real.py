from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
def _put_char_real(self, char):
    font = self.fonts[self.f]
    if font._vf is None:
        self.text.append(Text(self.h, self.v, font, char, font._width_of(char)))
    else:
        scale = font._scale
        for x, y, f, g, w in font._vf[char].text:
            newf = DviFont(scale=_mul2012(scale, f._scale), tfm=f._tfm, texname=f.texname, vf=f._vf)
            self.text.append(Text(self.h + _mul2012(x, scale), self.v + _mul2012(y, scale), newf, g, newf._width_of(g)))
        self.boxes.extend([Box(self.h + _mul2012(x, scale), self.v + _mul2012(y, scale), _mul2012(a, scale), _mul2012(b, scale)) for x, y, a, b in font._vf[char].boxes])