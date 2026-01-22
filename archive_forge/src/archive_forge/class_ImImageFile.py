from __future__ import annotations
import os
import re
from . import Image, ImageFile, ImagePalette
class ImImageFile(ImageFile.ImageFile):
    format = 'IM'
    format_description = 'IFUNC Image Memory'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if b'\n' not in self.fp.read(100):
            msg = 'not an IM file'
            raise SyntaxError(msg)
        self.fp.seek(0)
        n = 0
        self.info[MODE] = 'L'
        self.info[SIZE] = (512, 512)
        self.info[FRAMES] = 1
        self.rawmode = 'L'
        while True:
            s = self.fp.read(1)
            if s == b'\r':
                continue
            if not s or s == b'\x00' or s == b'\x1a':
                break
            s = s + self.fp.readline()
            if len(s) > 100:
                msg = 'not an IM file'
                raise SyntaxError(msg)
            if s[-2:] == b'\r\n':
                s = s[:-2]
            elif s[-1:] == b'\n':
                s = s[:-1]
            try:
                m = split.match(s)
            except re.error as e:
                msg = 'not an IM file'
                raise SyntaxError(msg) from e
            if m:
                k, v = m.group(1, 2)
                k = k.decode('latin-1', 'replace')
                v = v.decode('latin-1', 'replace')
                if k in [FRAMES, SCALE, SIZE]:
                    v = v.replace('*', ',')
                    v = tuple(map(number, v.split(',')))
                    if len(v) == 1:
                        v = v[0]
                elif k == MODE and v in OPEN:
                    v, self.rawmode = OPEN[v]
                if k == COMMENT:
                    if k in self.info:
                        self.info[k].append(v)
                    else:
                        self.info[k] = [v]
                else:
                    self.info[k] = v
                if k in TAGS:
                    n += 1
            else:
                msg = 'Syntax error in IM header: ' + s.decode('ascii', 'replace')
                raise SyntaxError(msg)
        if not n:
            msg = 'Not an IM file'
            raise SyntaxError(msg)
        self._size = self.info[SIZE]
        self._mode = self.info[MODE]
        while s and s[:1] != b'\x1a':
            s = self.fp.read(1)
        if not s:
            msg = 'File truncated'
            raise SyntaxError(msg)
        if LUT in self.info:
            palette = self.fp.read(768)
            greyscale = 1
            linear = 1
            for i in range(256):
                if palette[i] == palette[i + 256] == palette[i + 512]:
                    if palette[i] != i:
                        linear = 0
                else:
                    greyscale = 0
            if self.mode in ['L', 'LA', 'P', 'PA']:
                if greyscale:
                    if not linear:
                        self.lut = list(palette[:256])
                else:
                    if self.mode in ['L', 'P']:
                        self._mode = self.rawmode = 'P'
                    elif self.mode in ['LA', 'PA']:
                        self._mode = 'PA'
                        self.rawmode = 'PA;L'
                    self.palette = ImagePalette.raw('RGB;L', palette)
            elif self.mode == 'RGB':
                if not greyscale or not linear:
                    self.lut = list(palette)
        self.frame = 0
        self.__offset = offs = self.fp.tell()
        self._fp = self.fp
        if self.rawmode[:2] == 'F;':
            try:
                bits = int(self.rawmode[2:])
                if bits not in [8, 16, 32]:
                    self.tile = [('bit', (0, 0) + self.size, offs, (bits, 8, 3, 0, -1))]
                    return
            except ValueError:
                pass
        if self.rawmode in ['RGB;T', 'RYB;T']:
            size = self.size[0] * self.size[1]
            self.tile = [('raw', (0, 0) + self.size, offs, ('G', 0, -1)), ('raw', (0, 0) + self.size, offs + size, ('R', 0, -1)), ('raw', (0, 0) + self.size, offs + 2 * size, ('B', 0, -1))]
        else:
            self.tile = [('raw', (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    @property
    def n_frames(self):
        return self.info[FRAMES]

    @property
    def is_animated(self):
        return self.info[FRAMES] > 1

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.frame = frame
        if self.mode == '1':
            bits = 1
        else:
            bits = 8 * len(self.mode)
        size = (self.size[0] * bits + 7) // 8 * self.size[1]
        offs = self.__offset + frame * size
        self.fp = self._fp
        self.tile = [('raw', (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    def tell(self):
        return self.frame