from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DdsImageFile(ImageFile.ImageFile):
    format = 'DDS'
    format_description = 'DirectDraw Surface'

    def _open(self):
        if not _accept(self.fp.read(4)):
            msg = 'not a DDS file'
            raise SyntaxError(msg)
        header_size, = struct.unpack('<I', self.fp.read(4))
        if header_size != 124:
            msg = f'Unsupported header size {repr(header_size)}'
            raise OSError(msg)
        header_bytes = self.fp.read(header_size - 4)
        if len(header_bytes) != 120:
            msg = f'Incomplete header: {len(header_bytes)} bytes'
            raise OSError(msg)
        header = io.BytesIO(header_bytes)
        flags, height, width = struct.unpack('<3I', header.read(12))
        self._size = (width, height)
        extents = (0, 0) + self.size
        pitch, depth, mipmaps = struct.unpack('<3I', header.read(12))
        struct.unpack('<11I', header.read(44))
        pfsize, pfflags, fourcc, bitcount = struct.unpack('<4I', header.read(16))
        n = 0
        rawmode = None
        if pfflags & DDPF.RGB:
            if pfflags & DDPF.ALPHAPIXELS:
                self._mode = 'RGBA'
                mask_count = 4
            else:
                self._mode = 'RGB'
                mask_count = 3
            masks = struct.unpack(f'<{mask_count}I', header.read(mask_count * 4))
            self.tile = [('dds_rgb', extents, 0, (bitcount, masks))]
            return
        elif pfflags & DDPF.LUMINANCE:
            if bitcount == 8:
                self._mode = 'L'
            elif bitcount == 16 and pfflags & DDPF.ALPHAPIXELS:
                self._mode = 'LA'
            else:
                msg = f'Unsupported bitcount {bitcount} for {pfflags}'
                raise OSError(msg)
        elif pfflags & DDPF.PALETTEINDEXED8:
            self._mode = 'P'
            self.palette = ImagePalette.raw('RGBA', self.fp.read(1024))
        elif pfflags & DDPF.FOURCC:
            offset = header_size + 4
            if fourcc == D3DFMT.DXT1:
                self._mode = 'RGBA'
                self.pixel_format = 'DXT1'
                n = 1
            elif fourcc == D3DFMT.DXT3:
                self._mode = 'RGBA'
                self.pixel_format = 'DXT3'
                n = 2
            elif fourcc == D3DFMT.DXT5:
                self._mode = 'RGBA'
                self.pixel_format = 'DXT5'
                n = 3
            elif fourcc in (D3DFMT.BC4U, D3DFMT.ATI1):
                self._mode = 'L'
                self.pixel_format = 'BC4'
                n = 4
            elif fourcc == D3DFMT.BC5S:
                self._mode = 'RGB'
                self.pixel_format = 'BC5S'
                n = 5
            elif fourcc in (D3DFMT.BC5U, D3DFMT.ATI2):
                self._mode = 'RGB'
                self.pixel_format = 'BC5'
                n = 5
            elif fourcc == D3DFMT.DX10:
                offset += 20
                dxgi_format, = struct.unpack('<I', self.fp.read(4))
                self.fp.read(16)
                if dxgi_format in (DXGI_FORMAT.BC1_UNORM, DXGI_FORMAT.BC1_TYPELESS):
                    self._mode = 'RGBA'
                    self.pixel_format = 'BC1'
                    n = 1
                elif dxgi_format in (DXGI_FORMAT.BC4_TYPELESS, DXGI_FORMAT.BC4_UNORM):
                    self._mode = 'L'
                    self.pixel_format = 'BC4'
                    n = 4
                elif dxgi_format in (DXGI_FORMAT.BC5_TYPELESS, DXGI_FORMAT.BC5_UNORM):
                    self._mode = 'RGB'
                    self.pixel_format = 'BC5'
                    n = 5
                elif dxgi_format == DXGI_FORMAT.BC5_SNORM:
                    self._mode = 'RGB'
                    self.pixel_format = 'BC5S'
                    n = 5
                elif dxgi_format == DXGI_FORMAT.BC6H_UF16:
                    self._mode = 'RGB'
                    self.pixel_format = 'BC6H'
                    n = 6
                elif dxgi_format == DXGI_FORMAT.BC6H_SF16:
                    self._mode = 'RGB'
                    self.pixel_format = 'BC6HS'
                    n = 6
                elif dxgi_format in (DXGI_FORMAT.BC7_TYPELESS, DXGI_FORMAT.BC7_UNORM, DXGI_FORMAT.BC7_UNORM_SRGB):
                    self._mode = 'RGBA'
                    self.pixel_format = 'BC7'
                    n = 7
                    if dxgi_format == DXGI_FORMAT.BC7_UNORM_SRGB:
                        self.info['gamma'] = 1 / 2.2
                elif dxgi_format in (DXGI_FORMAT.R8G8B8A8_TYPELESS, DXGI_FORMAT.R8G8B8A8_UNORM, DXGI_FORMAT.R8G8B8A8_UNORM_SRGB):
                    self._mode = 'RGBA'
                    if dxgi_format == DXGI_FORMAT.R8G8B8A8_UNORM_SRGB:
                        self.info['gamma'] = 1 / 2.2
                else:
                    msg = f'Unimplemented DXGI format {dxgi_format}'
                    raise NotImplementedError(msg)
            else:
                msg = f'Unimplemented pixel format {repr(fourcc)}'
                raise NotImplementedError(msg)
        else:
            msg = f'Unknown pixel format flags {pfflags}'
            raise NotImplementedError(msg)
        if n:
            self.tile = [ImageFile._Tile('bcn', extents, offset, (n, self.pixel_format))]
        else:
            self.tile = [ImageFile._Tile('raw', extents, 0, rawmode or self.mode)]

    def load_seek(self, pos):
        pass