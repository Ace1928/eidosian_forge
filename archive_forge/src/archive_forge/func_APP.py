from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def APP(self, marker):
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    app = 'APP%d' % (marker & 15)
    self.app[app] = s
    self.applist.append((app, s))
    if marker == 65504 and s[:4] == b'JFIF':
        self.info['jfif'] = version = i16(s, 5)
        self.info['jfif_version'] = divmod(version, 256)
        try:
            jfif_unit = s[7]
            jfif_density = (i16(s, 8), i16(s, 10))
        except Exception:
            pass
        else:
            if jfif_unit == 1:
                self.info['dpi'] = jfif_density
            self.info['jfif_unit'] = jfif_unit
            self.info['jfif_density'] = jfif_density
    elif marker == 65505 and s[:6] == b'Exif\x00\x00':
        if 'exif' in self.info:
            self.info['exif'] += s[6:]
        else:
            self.info['exif'] = s
            self._exif_offset = self.fp.tell() - n + 6
    elif marker == 65506 and s[:5] == b'FPXR\x00':
        self.info['flashpix'] = s
    elif marker == 65506 and s[:12] == b'ICC_PROFILE\x00':
        self.icclist.append(s)
    elif marker == 65517 and s[:14] == b'Photoshop 3.0\x00':
        offset = 14
        photoshop = self.info.setdefault('photoshop', {})
        while s[offset:offset + 4] == b'8BIM':
            try:
                offset += 4
                code = i16(s, offset)
                offset += 2
                name_len = s[offset]
                offset += 1 + name_len
                offset += offset & 1
                size = i32(s, offset)
                offset += 4
                data = s[offset:offset + size]
                if code == 1005:
                    data = {'XResolution': i32(data, 0) / 65536, 'DisplayedUnitsX': i16(data, 4), 'YResolution': i32(data, 8) / 65536, 'DisplayedUnitsY': i16(data, 12)}
                photoshop[code] = data
                offset += size
                offset += offset & 1
            except struct.error:
                break
    elif marker == 65518 and s[:5] == b'Adobe':
        self.info['adobe'] = i16(s, 5)
        try:
            adobe_transform = s[11]
        except IndexError:
            pass
        else:
            self.info['adobe_transform'] = adobe_transform
    elif marker == 65506 and s[:4] == b'MPF\x00':
        self.info['mp'] = s[4:]
        self.info['mpoffset'] = self.fp.tell() - n + 4
    if 'dpi' not in self.info and 'exif' in self.info:
        try:
            exif = self.getexif()
            resolution_unit = exif[296]
            x_resolution = exif[282]
            try:
                dpi = float(x_resolution[0]) / x_resolution[1]
            except TypeError:
                dpi = x_resolution
            if math.isnan(dpi):
                msg = 'DPI is not a number'
                raise ValueError(msg)
            if resolution_unit == 3:
                dpi *= 2.54
            self.info['dpi'] = (dpi, dpi)
        except (struct.error, KeyError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
            self.info['dpi'] = (72, 72)