from __future__ import annotations
import io
from . import Image, ImageFile, ImagePalette
from ._binary import i8
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import si16be as si16
def _layerinfo(fp, ct_bytes):
    layers = []

    def read(size):
        return ImageFile._safe_read(fp, size)
    ct = si16(read(2))
    if ct_bytes < abs(ct) * 20:
        msg = 'Layer block too short for number of layers requested'
        raise SyntaxError(msg)
    for _ in range(abs(ct)):
        y0 = i32(read(4))
        x0 = i32(read(4))
        y1 = i32(read(4))
        x1 = i32(read(4))
        mode = []
        ct_types = i16(read(2))
        types = list(range(ct_types))
        if len(types) > 4:
            fp.seek(len(types) * 6 + 12, io.SEEK_CUR)
            size = i32(read(4))
            fp.seek(size, io.SEEK_CUR)
            continue
        for _ in types:
            type = i16(read(2))
            if type == 65535:
                m = 'A'
            else:
                m = 'RGBA'[type]
            mode.append(m)
            read(4)
        mode.sort()
        if mode == ['R']:
            mode = 'L'
        elif mode == ['B', 'G', 'R']:
            mode = 'RGB'
        elif mode == ['A', 'B', 'G', 'R']:
            mode = 'RGBA'
        else:
            mode = None
        read(12)
        name = ''
        size = i32(read(4))
        if size:
            data_end = fp.tell() + size
            length = i32(read(4))
            if length:
                fp.seek(length - 16, io.SEEK_CUR)
            length = i32(read(4))
            if length:
                fp.seek(length, io.SEEK_CUR)
            length = i8(read(1))
            if length:
                name = read(length).decode('latin-1', 'replace')
            fp.seek(data_end)
        layers.append((name, mode, (x0, y0, x1, y1)))
    for i, (name, mode, bbox) in enumerate(layers):
        tile = []
        for m in mode:
            t = _maketile(fp, m, bbox, 1)
            if t:
                tile.extend(t)
        layers[i] = (name, mode, bbox, tile)
    return layers