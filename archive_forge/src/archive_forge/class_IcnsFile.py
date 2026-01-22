from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
class IcnsFile:
    SIZES = {(512, 512, 2): [(b'ic10', read_png_or_jpeg2000)], (512, 512, 1): [(b'ic09', read_png_or_jpeg2000)], (256, 256, 2): [(b'ic14', read_png_or_jpeg2000)], (256, 256, 1): [(b'ic08', read_png_or_jpeg2000)], (128, 128, 2): [(b'ic13', read_png_or_jpeg2000)], (128, 128, 1): [(b'ic07', read_png_or_jpeg2000), (b'it32', read_32t), (b't8mk', read_mk)], (64, 64, 1): [(b'icp6', read_png_or_jpeg2000)], (32, 32, 2): [(b'ic12', read_png_or_jpeg2000)], (48, 48, 1): [(b'ih32', read_32), (b'h8mk', read_mk)], (32, 32, 1): [(b'icp5', read_png_or_jpeg2000), (b'il32', read_32), (b'l8mk', read_mk)], (16, 16, 2): [(b'ic11', read_png_or_jpeg2000)], (16, 16, 1): [(b'icp4', read_png_or_jpeg2000), (b'is32', read_32), (b's8mk', read_mk)]}

    def __init__(self, fobj):
        """
        fobj is a file-like object as an icns resource
        """
        self.dct = dct = {}
        self.fobj = fobj
        sig, filesize = nextheader(fobj)
        if not _accept(sig):
            msg = 'not an icns file'
            raise SyntaxError(msg)
        i = HEADERSIZE
        while i < filesize:
            sig, blocksize = nextheader(fobj)
            if blocksize <= 0:
                msg = 'invalid block header'
                raise SyntaxError(msg)
            i += HEADERSIZE
            blocksize -= HEADERSIZE
            dct[sig] = (i, blocksize)
            fobj.seek(blocksize, io.SEEK_CUR)
            i += blocksize

    def itersizes(self):
        sizes = []
        for size, fmts in self.SIZES.items():
            for fmt, reader in fmts:
                if fmt in self.dct:
                    sizes.append(size)
                    break
        return sizes

    def bestsize(self):
        sizes = self.itersizes()
        if not sizes:
            msg = 'No 32bit icon resources found'
            raise SyntaxError(msg)
        return max(sizes)

    def dataforsize(self, size):
        """
        Get an icon resource as {channel: array}.  Note that
        the arrays are bottom-up like windows bitmaps and will likely
        need to be flipped or transposed in some way.
        """
        dct = {}
        for code, reader in self.SIZES[size]:
            desc = self.dct.get(code)
            if desc is not None:
                dct.update(reader(self.fobj, desc, size))
        return dct

    def getimage(self, size=None):
        if size is None:
            size = self.bestsize()
        if len(size) == 2:
            size = (size[0], size[1], 1)
        channels = self.dataforsize(size)
        im = channels.get('RGBA', None)
        if im:
            return im
        im = channels.get('RGB').copy()
        try:
            im.putalpha(channels['A'])
        except KeyError:
            pass
        return im