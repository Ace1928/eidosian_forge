from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def _encode_tile(im, fp, tile: list[_Tile], bufsize, fh, exc=None):
    for encoder_name, extents, offset, args in tile:
        if offset > 0:
            fp.seek(offset)
        encoder = Image._getencoder(im.mode, encoder_name, args, im.encoderconfig)
        try:
            encoder.setimage(im.im, extents)
            if encoder.pushes_fd:
                encoder.setfd(fp)
                errcode = encoder.encode_to_pyfd()[1]
            elif exc:
                while True:
                    errcode, data = encoder.encode(bufsize)[1:]
                    fp.write(data)
                    if errcode:
                        break
            else:
                errcode = encoder.encode_to_file(fh, bufsize)
            if errcode < 0:
                raise _get_oserror(errcode, encoder=True) from exc
        finally:
            encoder.cleanup()