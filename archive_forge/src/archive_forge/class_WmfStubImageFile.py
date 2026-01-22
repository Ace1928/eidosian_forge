from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long
class WmfStubImageFile(ImageFile.StubImageFile):
    format = 'WMF'
    format_description = 'Windows Metafile'

    def _open(self):
        self._inch = None
        s = self.fp.read(80)
        if s[:6] == b'\xd7\xcd\xc6\x9a\x00\x00':
            self._inch = word(s, 14)
            x0 = short(s, 6)
            y0 = short(s, 8)
            x1 = short(s, 10)
            y1 = short(s, 12)
            self.info['dpi'] = 72
            size = ((x1 - x0) * self.info['dpi'] // self._inch, (y1 - y0) * self.info['dpi'] // self._inch)
            self.info['wmf_bbox'] = (x0, y0, x1, y1)
            if s[22:26] != b'\x01\x00\t\x00':
                msg = 'Unsupported WMF file format'
                raise SyntaxError(msg)
        elif s[:4] == b'\x01\x00\x00\x00' and s[40:44] == b' EMF':
            x0 = _long(s, 8)
            y0 = _long(s, 12)
            x1 = _long(s, 16)
            y1 = _long(s, 20)
            frame = (_long(s, 24), _long(s, 28), _long(s, 32), _long(s, 36))
            size = (x1 - x0, y1 - y0)
            xdpi = 2540.0 * (x1 - y0) / (frame[2] - frame[0])
            ydpi = 2540.0 * (y1 - y0) / (frame[3] - frame[1])
            self.info['wmf_bbox'] = (x0, y0, x1, y1)
            if xdpi == ydpi:
                self.info['dpi'] = xdpi
            else:
                self.info['dpi'] = (xdpi, ydpi)
        else:
            msg = 'Unsupported file format'
            raise SyntaxError(msg)
        self._mode = 'RGB'
        self._size = size
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler

    def load(self, dpi=None):
        if dpi is not None and self._inch is not None:
            self.info['dpi'] = dpi
            x0, y0, x1, y1 = self.info['wmf_bbox']
            self._size = ((x1 - x0) * self.info['dpi'] // self._inch, (y1 - y0) * self.info['dpi'] // self._inch)
        return super().load()