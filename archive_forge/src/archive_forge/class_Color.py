import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class Color:
    """This class is used to represent color.  Components red, green, blue
    are in the range 0 (dark) to 1 (full intensity)."""

    def __init__(self, red=0, green=0, blue=0, alpha=1):
        """Initialize with red, green, blue in range [0-1]."""
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __repr__(self):
        return 'Color(%s)' % fp_str(*(self.red, self.green, self.blue, self.alpha)).replace(' ', ',')

    @property
    def __key__(self):
        """simple comparison by component; cmyk != color ever
        >>> from reportlab import cmp
        >>> cmp(Color(0,0,0),None)
        -1
        >>> cmp(Color(0,0,0),black)
        0
        >>> cmp(Color(0,0,0),CMYKColor(0,0,0,1)),Color(0,0,0).rgba()==CMYKColor(0,0,0,1).rgba()
        (1, True)
        """
        return (self.red, self.green, self.blue, self.alpha)

    def __hash__(self):
        return hash(self.__key__)

    def __comparable__(self, other):
        return not isinstance(other, CMYKColor) and isinstance(other, Color)

    def __lt__(self, other):
        if not self.__comparable__(other):
            return True
        try:
            return self.__key__ < other.__key__
        except:
            pass
        return True

    def __eq__(self, other):
        if not self.__comparable__(other):
            return False
        try:
            return self.__key__ == other.__key__
        except:
            return False

    def rgb(self):
        """Returns a three-tuple of components"""
        return (self.red, self.green, self.blue)

    def rgba(self):
        """Returns a four-tuple of components"""
        return (self.red, self.green, self.blue, self.alpha)

    def bitmap_rgb(self):
        return tuple([int(x * 255) & 255 for x in self.rgb()])

    def bitmap_rgba(self):
        return tuple([int(x * 255) & 255 for x in self.rgba()])

    def hexval(self):
        return '0x%02x%02x%02x' % self.bitmap_rgb()

    def hexvala(self):
        return '0x%02x%02x%02x%02x' % self.bitmap_rgba()

    def int_rgb(self):
        v = self.bitmap_rgb()
        return v[0] << 16 | v[1] << 8 | v[2]

    def int_rgba(self):
        v = self.bitmap_rgba()
        return int((v[0] << 24 | v[1] << 16 | v[2] << 8 | v[3]) & 16777215)
    _cKwds = 'red green blue alpha'.split()

    def cKwds(self):
        for k in self._cKwds:
            yield (k, getattr(self, k))
    cKwds = property(cKwds)

    def clone(self, **kwds):
        """copy then change values in kwds"""
        D = dict([kv for kv in self.cKwds])
        D.update(kwds)
        return self.__class__(**D)

    def _lookupName(self, D={}):
        if not D:
            for n, v in getAllNamedColors().items():
                if not isinstance(v, CMYKColor):
                    t = (v.red, v.green, v.blue)
                    if t in D:
                        n = n + '/' + D[t]
                    D[t] = n
        t = (self.red, self.green, self.blue)
        return t in D and D[t] or None

    @property
    def normalizedAlpha(self):
        return self.alpha