from __future__ import annotations
import io
from . import Image, ImageFile, ImagePalette
from ._binary import i8
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import si16be as si16
class PsdImageFile(ImageFile.ImageFile):
    format = 'PSD'
    format_description = 'Adobe Photoshop'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        read = self.fp.read
        s = read(26)
        if not _accept(s) or i16(s, 4) != 1:
            msg = 'not a PSD file'
            raise SyntaxError(msg)
        psd_bits = i16(s, 22)
        psd_channels = i16(s, 12)
        psd_mode = i16(s, 24)
        mode, channels = MODES[psd_mode, psd_bits]
        if channels > psd_channels:
            msg = 'not enough channels'
            raise OSError(msg)
        if mode == 'RGB' and psd_channels == 4:
            mode = 'RGBA'
            channels = 4
        self._mode = mode
        self._size = (i32(s, 18), i32(s, 14))
        size = i32(read(4))
        if size:
            data = read(size)
            if mode == 'P' and size == 768:
                self.palette = ImagePalette.raw('RGB;L', data)
        self.resources = []
        size = i32(read(4))
        if size:
            end = self.fp.tell() + size
            while self.fp.tell() < end:
                read(4)
                id = i16(read(2))
                name = read(i8(read(1)))
                if not len(name) & 1:
                    read(1)
                data = read(i32(read(4)))
                if len(data) & 1:
                    read(1)
                self.resources.append((id, name, data))
                if id == 1039:
                    self.info['icc_profile'] = data
        self.layers = []
        size = i32(read(4))
        if size:
            end = self.fp.tell() + size
            size = i32(read(4))
            if size:
                _layer_data = io.BytesIO(ImageFile._safe_read(self.fp, size))
                self.layers = _layerinfo(_layer_data, size)
            self.fp.seek(end)
        self.n_frames = len(self.layers)
        self.is_animated = self.n_frames > 1
        self.tile = _maketile(self.fp, mode, (0, 0) + self.size, channels)
        self._fp = self.fp
        self.frame = 1
        self._min_frame = 1

    def seek(self, layer):
        if not self._seek_check(layer):
            return
        try:
            name, mode, bbox, tile = self.layers[layer - 1]
            self._mode = mode
            self.tile = tile
            self.frame = layer
            self.fp = self._fp
            return (name, bbox)
        except IndexError as e:
            msg = 'no such layer'
            raise EOFError(msg) from e

    def tell(self):
        return self.frame