from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
class GlyphCoordinates(object):
    """A list of glyph coordinates.

    Unlike an ordinary list, this is a numpy-like matrix object which supports
    matrix addition, scalar multiplication and other operations described below.
    """

    def __init__(self, iterable=[]):
        self._a = array.array('d')
        self.extend(iterable)

    @property
    def array(self):
        """Returns the underlying array of coordinates"""
        return self._a

    @staticmethod
    def zeros(count):
        """Creates a new ``GlyphCoordinates`` object with all coordinates set to (0,0)"""
        g = GlyphCoordinates()
        g._a.frombytes(bytes(count * 2 * g._a.itemsize))
        return g

    def copy(self):
        """Creates a new ``GlyphCoordinates`` object which is a copy of the current one."""
        c = GlyphCoordinates()
        c._a.extend(self._a)
        return c

    def __len__(self):
        """Returns the number of coordinates in the array."""
        return len(self._a) // 2

    def __getitem__(self, k):
        """Returns a two element tuple (x,y)"""
        a = self._a
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            ret = []
            for k in indices:
                x = a[2 * k]
                y = a[2 * k + 1]
                ret.append((int(x) if x.is_integer() else x, int(y) if y.is_integer() else y))
            return ret
        x = a[2 * k]
        y = a[2 * k + 1]
        return (int(x) if x.is_integer() else x, int(y) if y.is_integer() else y)

    def __setitem__(self, k, v):
        """Sets a point's coordinates to a two element tuple (x,y)"""
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            for j, i in enumerate(indices):
                self[i] = v[j]
            return
        self._a[2 * k], self._a[2 * k + 1] = v

    def __delitem__(self, i):
        """Removes a point from the list"""
        i = 2 * i % len(self._a)
        del self._a[i]
        del self._a[i]

    def __repr__(self):
        return 'GlyphCoordinates([' + ','.join((str(c) for c in self)) + '])'

    def append(self, p):
        self._a.extend(tuple(p))

    def extend(self, iterable):
        for p in iterable:
            self._a.extend(p)

    def toInt(self, *, round=otRound):
        if round is noRound:
            return
        a = self._a
        for i in range(len(a)):
            a[i] = round(a[i])

    def calcBounds(self):
        a = self._a
        if not a:
            return (0, 0, 0, 0)
        xs = a[0::2]
        ys = a[1::2]
        return (min(xs), min(ys), max(xs), max(ys))

    def calcIntBounds(self, round=otRound):
        return tuple((round(v) for v in self.calcBounds()))

    def relativeToAbsolute(self):
        a = self._a
        x, y = (0, 0)
        for i in range(0, len(a), 2):
            a[i] = x = a[i] + x
            a[i + 1] = y = a[i + 1] + y

    def absoluteToRelative(self):
        a = self._a
        x, y = (0, 0)
        for i in range(0, len(a), 2):
            nx = a[i]
            ny = a[i + 1]
            a[i] = nx - x
            a[i + 1] = ny - y
            x = nx
            y = ny

    def translate(self, p):
        """
        >>> GlyphCoordinates([(1,2)]).translate((.5,0))
        """
        x, y = p
        if x == 0 and y == 0:
            return
        a = self._a
        for i in range(0, len(a), 2):
            a[i] += x
            a[i + 1] += y

    def scale(self, p):
        """
        >>> GlyphCoordinates([(1,2)]).scale((.5,0))
        """
        x, y = p
        if x == 1 and y == 1:
            return
        a = self._a
        for i in range(0, len(a), 2):
            a[i] *= x
            a[i + 1] *= y

    def transform(self, t):
        """
        >>> GlyphCoordinates([(1,2)]).transform(((.5,0),(.2,.5)))
        """
        a = self._a
        for i in range(0, len(a), 2):
            x = a[i]
            y = a[i + 1]
            px = x * t[0][0] + y * t[1][0]
            py = x * t[0][1] + y * t[1][1]
            a[i] = px
            a[i + 1] = py

    def __eq__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g2 = GlyphCoordinates([(1.0,2)])
        >>> g3 = GlyphCoordinates([(1.5,2)])
        >>> g == g2
        True
        >>> g == g3
        False
        >>> g2 == g3
        False
        """
        if type(self) != type(other):
            return NotImplemented
        return self._a == other._a

    def __ne__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g2 = GlyphCoordinates([(1.0,2)])
        >>> g3 = GlyphCoordinates([(1.5,2)])
        >>> g != g2
        False
        >>> g != g3
        True
        >>> g2 != g3
        True
        """
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __pos__(self):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        >>> g2 = +g
        >>> g2
        GlyphCoordinates([(1, 2)])
        >>> g2.translate((1,0))
        >>> g2
        GlyphCoordinates([(2, 2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        """
        return self.copy()

    def __neg__(self):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        >>> g2 = -g
        >>> g2
        GlyphCoordinates([(-1, -2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        """
        r = self.copy()
        a = r._a
        for i in range(len(a)):
            a[i] = -a[i]
        return r

    def __round__(self, *, round=otRound):
        r = self.copy()
        r.toInt(round=round)
        return r

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __truediv__(self, other):
        return self.copy().__itruediv__(other)
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + -self

    def __iadd__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g += (.5,0)
        >>> g
        GlyphCoordinates([(1.5, 2)])
        >>> g2 = GlyphCoordinates([(3,4)])
        >>> g += g2
        >>> g
        GlyphCoordinates([(4.5, 6)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.translate(other)
            return self
        if isinstance(other, GlyphCoordinates):
            other = other._a
            a = self._a
            assert len(a) == len(other)
            for i in range(len(a)):
                a[i] += other[i]
            return self
        return NotImplemented

    def __isub__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g -= (.5,0)
        >>> g
        GlyphCoordinates([(0.5, 2)])
        >>> g2 = GlyphCoordinates([(3,4)])
        >>> g -= g2
        >>> g
        GlyphCoordinates([(-2.5, -2)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.translate((-other[0], -other[1]))
            return self
        if isinstance(other, GlyphCoordinates):
            other = other._a
            a = self._a
            assert len(a) == len(other)
            for i in range(len(a)):
                a[i] -= other[i]
            return self
        return NotImplemented

    def __imul__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g *= (2,.5)
        >>> g *= 2
        >>> g
        GlyphCoordinates([(4, 2)])
        >>> g = GlyphCoordinates([(1,2)])
        >>> g *= 2
        >>> g
        GlyphCoordinates([(2, 4)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.scale(other)
            return self
        if isinstance(other, Number):
            if other == 1:
                return self
            a = self._a
            for i in range(len(a)):
                a[i] *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """
        >>> g = GlyphCoordinates([(1,3)])
        >>> g /= (.5,1.5)
        >>> g /= 2
        >>> g
        GlyphCoordinates([(1, 1)])
        """
        if isinstance(other, Number):
            other = (other, other)
        if isinstance(other, tuple):
            if other == (1, 1):
                return self
            assert len(other) == 2
            self.scale((1.0 / other[0], 1.0 / other[1]))
            return self
        return NotImplemented

    def __bool__(self):
        """
        >>> g = GlyphCoordinates([])
        >>> bool(g)
        False
        >>> g = GlyphCoordinates([(0,0), (0.,0)])
        >>> bool(g)
        True
        >>> g = GlyphCoordinates([(0,0), (1,0)])
        >>> bool(g)
        True
        >>> g = GlyphCoordinates([(0,.5), (0,0)])
        >>> bool(g)
        True
        """
        return bool(self._a)
    __nonzero__ = __bool__