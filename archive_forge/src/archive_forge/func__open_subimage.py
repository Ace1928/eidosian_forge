from __future__ import annotations
import olefile
from . import Image, ImageFile
from ._binary import i32le as i32
def _open_subimage(self, index=1, subimage=0):
    stream = [f'Data Object Store {index:06d}', f'Resolution {subimage:04d}', 'Subimage 0000 Header']
    fp = self.ole.openstream(stream)
    fp.read(28)
    s = fp.read(36)
    size = (i32(s, 4), i32(s, 8))
    tilesize = (i32(s, 16), i32(s, 20))
    offset = i32(s, 28)
    length = i32(s, 32)
    if size != self.size:
        msg = 'subimage mismatch'
        raise OSError(msg)
    fp.seek(28 + offset)
    s = fp.read(i32(s, 12) * length)
    x = y = 0
    xsize, ysize = size
    xtile, ytile = tilesize
    self.tile = []
    for i in range(0, len(s), length):
        x1 = min(xsize, x + xtile)
        y1 = min(ysize, y + ytile)
        compression = i32(s, i + 8)
        if compression == 0:
            self.tile.append(('raw', (x, y, x1, y1), i32(s, i) + 28, (self.rawmode,)))
        elif compression == 1:
            self.tile.append(('fill', (x, y, x1, y1), i32(s, i) + 28, (self.rawmode, s[12:16])))
        elif compression == 2:
            internal_color_conversion = s[14]
            jpeg_tables = s[15]
            rawmode = self.rawmode
            if internal_color_conversion:
                if rawmode == 'RGBA':
                    jpegmode, rawmode = ('YCbCrK', 'CMYK')
                else:
                    jpegmode = None
            else:
                jpegmode = rawmode
            self.tile.append(('jpeg', (x, y, x1, y1), i32(s, i) + 28, (rawmode, jpegmode)))
            if jpeg_tables:
                self.tile_prefix = self.jpeg[jpeg_tables]
        else:
            msg = 'unknown/invalid compression'
            raise OSError(msg)
        x = x + xtile
        if x >= xsize:
            x, y = (0, y + ytile)
            if y >= ysize:
                break
    self.stream = stream
    self._fp = self.fp
    self.fp = None