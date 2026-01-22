from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def _parse_jp2_header(fp):
    """Parse the JP2 header box to extract size, component count,
    color space information, and optionally DPI information,
    returning a (size, mode, mimetype, dpi) tuple."""
    reader = BoxReader(fp)
    header = None
    mimetype = None
    while reader.has_next_box():
        tbox = reader.next_box_type()
        if tbox == b'jp2h':
            header = reader.read_boxes()
            break
        elif tbox == b'ftyp':
            if reader.read_fields('>4s')[0] == b'jpx ':
                mimetype = 'image/jpx'
    size = None
    mode = None
    bpc = None
    nc = None
    dpi = None
    while header.has_next_box():
        tbox = header.next_box_type()
        if tbox == b'ihdr':
            height, width, nc, bpc = header.read_fields('>IIHB')
            size = (width, height)
            if nc == 1 and bpc & 127 > 8:
                mode = 'I;16'
            elif nc == 1:
                mode = 'L'
            elif nc == 2:
                mode = 'LA'
            elif nc == 3:
                mode = 'RGB'
            elif nc == 4:
                mode = 'RGBA'
        elif tbox == b'res ':
            res = header.read_boxes()
            while res.has_next_box():
                tres = res.next_box_type()
                if tres == b'resc':
                    vrcn, vrcd, hrcn, hrcd, vrce, hrce = res.read_fields('>HHHHBB')
                    hres = _res_to_dpi(hrcn, hrcd, hrce)
                    vres = _res_to_dpi(vrcn, vrcd, vrce)
                    if hres is not None and vres is not None:
                        dpi = (hres, vres)
                    break
    if size is None or mode is None:
        msg = 'Malformed JP2 header'
        raise SyntaxError(msg)
    return (size, mode, mimetype, dpi)