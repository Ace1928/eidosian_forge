import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class CMYKColor(Color):
    """This represents colors using the CMYK (cyan, magenta, yellow, black)
    model commonly used in professional printing.  This is implemented
    as a derived class so that renderers which only know about RGB "see it"
    as an RGB color through its 'red','green' and 'blue' attributes, according
    to an approximate function.

    The RGB approximation is worked out when the object in constructed, so
    the color attributes should not be changed afterwards.

    Extra attributes may be attached to the class to support specific ink models,
    and renderers may look for these."""
    _scale = 1.0

    def __init__(self, cyan=0, magenta=0, yellow=0, black=0, spotName=None, density=1, knockout=None, alpha=1):
        """
        Initialize with four colors in range [0-1]. the optional
        spotName, density & knockout may be of use to specific renderers.
        spotName is intended for use as an identifier to the renderer not client programs.
        density is used to modify the overall amount of ink.
        knockout is a renderer dependent option that determines whether the applied colour
        knocksout (removes) existing colour; None means use the global default.
        """
        self.cyan = cyan
        self.magenta = magenta
        self.yellow = yellow
        self.black = black
        self.spotName = spotName
        self.density = max(min(density, 1), 0)
        self.knockout = knockout
        self.alpha = alpha
        self.red, self.green, self.blue = cmyk2rgb((cyan, magenta, yellow, black))
        if density < 1:
            r, g, b = (self.red, self.green, self.blue)
            r = density * (r - 1) + 1
            g = density * (g - 1) + 1
            b = density * (b - 1) + 1
            self.red, self.green, self.blue = (r, g, b)

    def __repr__(self):
        return '%s(%s%s%s%s%s)' % (self.__class__.__name__, fp_str(self.cyan, self.magenta, self.yellow, self.black).replace(' ', ','), self.spotName and ',spotName=' + repr(self.spotName) or '', self.density != 1 and ',density=' + fp_str(self.density) or '', self.knockout is not None and ',knockout=%d' % self.knockout or '', self.alpha is not None and ',alpha=%s' % self.alpha or '')

    def fader(self, n, reverse=False):
        """return n colors based on density fade
        *NB* note this dosen't reach density zero"""
        scale = self._scale
        dd = scale / float(n)
        L = [self.clone(density=scale - i * dd) for i in range(n)]
        if reverse:
            L.reverse()
        return L

    @property
    def __key__(self):
        """obvious way to compare colours
        Comparing across the two color models is of limited use.
        >>> cmp(CMYKColor(0,0,0,1),None)
        -1
        >>> cmp(CMYKColor(0,0,0,1),_CMYK_black)
        0
        >>> cmp(PCMYKColor(0,0,0,100),_CMYK_black)
        0
        >>> cmp(CMYKColor(0,0,0,1),Color(0,0,1)),Color(0,0,0).rgba()==CMYKColor(0,0,0,1).rgba()
        (-1, True)
        """
        return (self.cyan, self.magenta, self.yellow, self.black, self.density, self.spotName, self.alpha)

    def __comparable__(self, other):
        return isinstance(other, CMYKColor)

    def cmyk(self):
        """Returns a tuple of four color components - syntactic sugar"""
        return (self.cyan, self.magenta, self.yellow, self.black)

    def cmyka(self):
        """Returns a tuple of five color components - syntactic sugar"""
        return (self.cyan, self.magenta, self.yellow, self.black, self.alpha)

    def _density_str(self):
        return fp_str(self.density)
    _cKwds = 'cyan magenta yellow black density alpha spotName knockout'.split()

    def _lookupName(self, D={}):
        if not D:
            for n, v in getAllNamedColors().items():
                if isinstance(v, CMYKColor):
                    t = (v.cyan, v.magenta, v.yellow, v.black)
                    if t in D:
                        n = n + '/' + D[t]
                    D[t] = n
        t = (self.cyan, self.magenta, self.yellow, self.black)
        return t in D and D[t] or None

    @property
    def normalizedAlpha(self):
        return self.alpha * self._scale