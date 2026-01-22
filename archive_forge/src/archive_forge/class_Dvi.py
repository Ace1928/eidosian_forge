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
class Dvi:
    """
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
        ...     for page in dvi:
        ...         print(''.join(chr(t.glyph) for t in page.text))
    """
    _dtable = [None] * 256
    _dispatch = partial(_dispatch, _dtable)

    def __init__(self, filename, dpi):
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
        _log.debug('Dvi: %s', filename)
        self.file = open(filename, 'rb')
        self.dpi = dpi
        self.fonts = {}
        self.state = _dvistate.pre

    def __enter__(self):
        """Context manager enter method, does nothing."""
        return self

    def __exit__(self, etype, evalue, etrace):
        """
        Context manager exit method, closes the underlying file if it is open.
        """
        self.close()

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Yields
        ------
        Page
            Details of all the text and box objects on the page.
            The Page tuple contains lists of Text and Box tuples and
            the page dimensions, and the Text and Box tuples contain
            coordinates transformed into a standard Cartesian
            coordinate system at the dpi value given when initializing.
            The coordinates are floating point numbers, but otherwise
            precision is not lost and coordinate values are not clipped to
            integers.
        """
        while self._read():
            yield self._output()

    def close(self):
        """Close the underlying file if it is open."""
        if not self.file.closed:
            self.file.close()

    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
        minx, miny, maxx, maxy = (np.inf, np.inf, -np.inf, -np.inf)
        maxy_pure = -np.inf
        for elt in self.text + self.boxes:
            if isinstance(elt, Box):
                x, y, h, w = elt
                e = 0
            else:
                x, y, font, g, w = elt
                h, e = font._height_depth_of(g)
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)
        if self._baseline_v is not None:
            maxy_pure = self._baseline_v
            self._baseline_v = None
        if not self.text and (not self.boxes):
            return Page(text=[], boxes=[], width=0, height=0, descent=0)
        if self.dpi is None:
            return Page(text=self.text, boxes=self.boxes, width=maxx - minx, height=maxy_pure - miny, descent=maxy - maxy_pure)
        d = self.dpi / (72.27 * 2 ** 16)
        descent = (maxy - maxy_pure) * d
        text = [Text((x - minx) * d, (maxy - y) * d - descent, f, g, w * d) for x, y, f, g, w in self.text]
        boxes = [Box((x - minx) * d, (maxy - y) * d - descent, h * d, w * d) for x, y, h, w in self.boxes]
        return Page(text=text, boxes=boxes, width=(maxx - minx) * d, height=(maxy_pure - miny) * d, descent=descent)

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        down_stack = [0]
        self._baseline_v = None
        while True:
            byte = self.file.read(1)[0]
            self._dtable[byte](self, byte)
            name = self._dtable[byte].__name__
            if name == '_push':
                down_stack.append(down_stack[-1])
            elif name == '_pop':
                down_stack.pop()
            elif name == '_down':
                down_stack[-1] += 1
            if self._baseline_v is None and len(getattr(self, 'stack', [])) == 3 and (down_stack[-1] >= 4):
                self._baseline_v = self.v
            if byte == 140:
                return True
            if self.state is _dvistate.post_post:
                self.close()
                return False

    def _arg(self, nbytes, signed=False):
        """
        Read and return an integer argument *nbytes* long.
        Signedness is determined by the *signed* keyword.
        """
        buf = self.file.read(nbytes)
        value = buf[0]
        if signed and value >= 128:
            value = value - 256
        for b in buf[1:]:
            value = 256 * value + b
        return value

    @_dispatch(min=0, max=127, state=_dvistate.inpage)
    def _set_char_immediate(self, char):
        self._put_char_real(char)
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(min=128, max=131, state=_dvistate.inpage, args=('olen1',))
    def _set_char(self, char):
        self._put_char_real(char)
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(132, state=_dvistate.inpage, args=('s4', 's4'))
    def _set_rule(self, a, b):
        self._put_rule_real(a, b)
        self.h += b

    @_dispatch(min=133, max=136, state=_dvistate.inpage, args=('olen1',))
    def _put_char(self, char):
        self._put_char_real(char)

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

    @_dispatch(137, state=_dvistate.inpage, args=('s4', 's4'))
    def _put_rule(self, a, b):
        self._put_rule_real(a, b)

    def _put_rule_real(self, a, b):
        if a > 0 and b > 0:
            self.boxes.append(Box(self.h, self.v, a, b))

    @_dispatch(138)
    def _nop(self, _):
        pass

    @_dispatch(139, state=_dvistate.outer, args=('s4',) * 11)
    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        self.state = _dvistate.inpage
        self.h, self.v, self.w, self.x, self.y, self.z = (0, 0, 0, 0, 0, 0)
        self.stack = []
        self.text = []
        self.boxes = []

    @_dispatch(140, state=_dvistate.inpage)
    def _eop(self, _):
        self.state = _dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    @_dispatch(141, state=_dvistate.inpage)
    def _push(self, _):
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    @_dispatch(142, state=_dvistate.inpage)
    def _pop(self, _):
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    @_dispatch(min=143, max=146, state=_dvistate.inpage, args=('slen1',))
    def _right(self, b):
        self.h += b

    @_dispatch(min=147, max=151, state=_dvistate.inpage, args=('slen',))
    def _right_w(self, new_w):
        if new_w is not None:
            self.w = new_w
        self.h += self.w

    @_dispatch(min=152, max=156, state=_dvistate.inpage, args=('slen',))
    def _right_x(self, new_x):
        if new_x is not None:
            self.x = new_x
        self.h += self.x

    @_dispatch(min=157, max=160, state=_dvistate.inpage, args=('slen1',))
    def _down(self, a):
        self.v += a

    @_dispatch(min=161, max=165, state=_dvistate.inpage, args=('slen',))
    def _down_y(self, new_y):
        if new_y is not None:
            self.y = new_y
        self.v += self.y

    @_dispatch(min=166, max=170, state=_dvistate.inpage, args=('slen',))
    def _down_z(self, new_z):
        if new_z is not None:
            self.z = new_z
        self.v += self.z

    @_dispatch(min=171, max=234, state=_dvistate.inpage)
    def _fnt_num_immediate(self, k):
        self.f = k

    @_dispatch(min=235, max=238, state=_dvistate.inpage, args=('olen1',))
    def _fnt_num(self, new_f):
        self.f = new_f

    @_dispatch(min=239, max=242, args=('ulen1',))
    def _xxx(self, datalen):
        special = self.file.read(datalen)
        _log.debug('Dvi._xxx: encountered special: %s', ''.join([chr(ch) if 32 <= ch < 127 else '<%02x>' % ch for ch in special]))

    @_dispatch(min=243, max=246, args=('olen1', 'u4', 'u4', 'u4', 'u1', 'u1'))
    def _fnt_def(self, k, c, s, d, a, l):
        self._fnt_def_real(k, c, s, d, a, l)

    def _fnt_def_real(self, k, c, s, d, a, l):
        n = self.file.read(a + l)
        fontname = n[-l:].decode('ascii')
        tfm = _tfmfile(fontname)
        if c != 0 and tfm.checksum != 0 and (c != tfm.checksum):
            raise ValueError('tfm checksum mismatch: %s' % n)
        try:
            vf = _vffile(fontname)
        except FileNotFoundError:
            vf = None
        self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)

    @_dispatch(247, state=_dvistate.pre, args=('u1', 'u4', 'u4', 'u4', 'u1'))
    def _pre(self, i, num, den, mag, k):
        self.file.read(k)
        if i != 2:
            raise ValueError('Unknown dvi format %d' % i)
        if num != 25400000 or den != 7227 * 2 ** 16:
            raise ValueError('Nonstandard units in dvi file')
        if mag != 1000:
            raise ValueError('Nonstandard magnification in dvi file')
        self.state = _dvistate.outer

    @_dispatch(248, state=_dvistate.outer)
    def _post(self, _):
        self.state = _dvistate.post_post

    @_dispatch(249)
    def _post_post(self, _):
        raise NotImplementedError

    @_dispatch(min=250, max=255)
    def _malformed(self, offset):
        raise ValueError(f'unknown command: byte {250 + offset}')