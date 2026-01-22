from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
class PpmImageFile(ImageFile.ImageFile):
    format = 'PPM'
    format_description = 'Pbmplus image'

    def _read_magic(self):
        magic = b''
        for _ in range(6):
            c = self.fp.read(1)
            if not c or c in b_whitespace:
                break
            magic += c
        return magic

    def _read_token(self):
        token = b''
        while len(token) <= 10:
            c = self.fp.read(1)
            if not c:
                break
            elif c in b_whitespace:
                if not token:
                    continue
                break
            elif c == b'#':
                while self.fp.read(1) not in b'\r\n':
                    pass
                continue
            token += c
        if not token:
            msg = 'Reached EOF while reading header'
            raise ValueError(msg)
        elif len(token) > 10:
            msg = f'Token too long in file header: {token.decode()}'
            raise ValueError(msg)
        return token

    def _open(self):
        magic_number = self._read_magic()
        try:
            mode = MODES[magic_number]
        except KeyError:
            msg = 'not a PPM file'
            raise SyntaxError(msg)
        if magic_number in (b'P1', b'P4'):
            self.custom_mimetype = 'image/x-portable-bitmap'
        elif magic_number in (b'P2', b'P5'):
            self.custom_mimetype = 'image/x-portable-graymap'
        elif magic_number in (b'P3', b'P6'):
            self.custom_mimetype = 'image/x-portable-pixmap'
        maxval = None
        decoder_name = 'raw'
        if magic_number in (b'P1', b'P2', b'P3'):
            decoder_name = 'ppm_plain'
        for ix in range(3):
            token = int(self._read_token())
            if ix == 0:
                xsize = token
            elif ix == 1:
                ysize = token
                if mode == '1':
                    self._mode = '1'
                    rawmode = '1;I'
                    break
                else:
                    self._mode = rawmode = mode
            elif ix == 2:
                maxval = token
                if not 0 < maxval < 65536:
                    msg = 'maxval must be greater than 0 and less than 65536'
                    raise ValueError(msg)
                if maxval > 255 and mode == 'L':
                    self._mode = 'I'
                if decoder_name != 'ppm_plain':
                    if maxval == 65535 and mode == 'L':
                        rawmode = 'I;16B'
                    elif maxval != 255:
                        decoder_name = 'ppm'
        args = (rawmode, 0, 1) if decoder_name == 'raw' else (rawmode, maxval)
        self._size = (xsize, ysize)
        self.tile = [(decoder_name, (0, 0, xsize, ysize), self.fp.tell(), args)]