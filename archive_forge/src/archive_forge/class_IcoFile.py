from __future__ import annotations
import warnings
from io import BytesIO
from math import ceil, log
from . import BmpImagePlugin, Image, ImageFile, PngImagePlugin
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
class IcoFile:

    def __init__(self, buf):
        """
        Parse image from file-like object containing ico file data
        """
        s = buf.read(6)
        if not _accept(s):
            msg = 'not an ICO file'
            raise SyntaxError(msg)
        self.buf = buf
        self.entry = []
        self.nb_items = i16(s, 4)
        for i in range(self.nb_items):
            s = buf.read(16)
            icon_header = {'width': s[0], 'height': s[1], 'nb_color': s[2], 'reserved': s[3], 'planes': i16(s, 4), 'bpp': i16(s, 6), 'size': i32(s, 8), 'offset': i32(s, 12)}
            for j in ('width', 'height'):
                if not icon_header[j]:
                    icon_header[j] = 256
            icon_header['color_depth'] = icon_header['bpp'] or (icon_header['nb_color'] != 0 and ceil(log(icon_header['nb_color'], 2))) or 256
            icon_header['dim'] = (icon_header['width'], icon_header['height'])
            icon_header['square'] = icon_header['width'] * icon_header['height']
            self.entry.append(icon_header)
        self.entry = sorted(self.entry, key=lambda x: x['color_depth'])
        self.entry = sorted(self.entry, key=lambda x: x['square'], reverse=True)

    def sizes(self):
        """
        Get a list of all available icon sizes and color depths.
        """
        return {(h['width'], h['height']) for h in self.entry}

    def getentryindex(self, size, bpp=False):
        for i, h in enumerate(self.entry):
            if size == h['dim'] and (bpp is False or bpp == h['color_depth']):
                return i
        return 0

    def getimage(self, size, bpp=False):
        """
        Get an image from the icon
        """
        return self.frame(self.getentryindex(size, bpp))

    def frame(self, idx):
        """
        Get an image from frame idx
        """
        header = self.entry[idx]
        self.buf.seek(header['offset'])
        data = self.buf.read(8)
        self.buf.seek(header['offset'])
        if data[:8] == PngImagePlugin._MAGIC:
            im = PngImagePlugin.PngImageFile(self.buf)
            Image._decompression_bomb_check(im.size)
        else:
            im = BmpImagePlugin.DibImageFile(self.buf)
            Image._decompression_bomb_check(im.size)
            im._size = (im.size[0], int(im.size[1] / 2))
            d, e, o, a = im.tile[0]
            im.tile[0] = (d, (0, 0) + im.size, o, a)
            bpp = header['bpp']
            if 32 == bpp:
                self.buf.seek(o)
                alpha_bytes = self.buf.read(im.size[0] * im.size[1] * 4)[3::4]
                mask = Image.frombuffer('L', im.size, alpha_bytes, 'raw', ('L', 0, -1))
            else:
                w = im.size[0]
                if w % 32 > 0:
                    w += 32 - im.size[0] % 32
                total_bytes = int(w * im.size[1] / 8)
                and_mask_offset = header['offset'] + header['size'] - total_bytes
                self.buf.seek(and_mask_offset)
                mask_data = self.buf.read(total_bytes)
                mask = Image.frombuffer('1', im.size, mask_data, 'raw', ('1;I', int(w / 8), -1))
            im = im.convert('RGBA')
            im.putalpha(mask)
        return im