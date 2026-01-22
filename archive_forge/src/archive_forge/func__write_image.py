from __future__ import annotations
import io
import math
import os
import time
from . import Image, ImageFile, ImageSequence, PdfParser, __version__, features
def _write_image(im, filename, existing_pdf, image_refs):
    params = None
    decode = None
    width, height = im.size
    dict_obj = {'BitsPerComponent': 8}
    if im.mode == '1':
        if features.check('libtiff'):
            filter = 'CCITTFaxDecode'
            dict_obj['BitsPerComponent'] = 1
            params = PdfParser.PdfArray([PdfParser.PdfDict({'K': -1, 'BlackIs1': True, 'Columns': width, 'Rows': height})])
        else:
            filter = 'DCTDecode'
        dict_obj['ColorSpace'] = PdfParser.PdfName('DeviceGray')
        procset = 'ImageB'
    elif im.mode == 'L':
        filter = 'DCTDecode'
        dict_obj['ColorSpace'] = PdfParser.PdfName('DeviceGray')
        procset = 'ImageB'
    elif im.mode == 'LA':
        filter = 'JPXDecode'
        procset = 'ImageB'
        dict_obj['SMaskInData'] = 1
    elif im.mode == 'P':
        filter = 'ASCIIHexDecode'
        palette = im.getpalette()
        dict_obj['ColorSpace'] = [PdfParser.PdfName('Indexed'), PdfParser.PdfName('DeviceRGB'), len(palette) // 3 - 1, PdfParser.PdfBinary(palette)]
        procset = 'ImageI'
        if 'transparency' in im.info:
            smask = im.convert('LA').getchannel('A')
            smask.encoderinfo = {}
            image_ref = _write_image(smask, filename, existing_pdf, image_refs)[0]
            dict_obj['SMask'] = image_ref
    elif im.mode == 'RGB':
        filter = 'DCTDecode'
        dict_obj['ColorSpace'] = PdfParser.PdfName('DeviceRGB')
        procset = 'ImageC'
    elif im.mode == 'RGBA':
        filter = 'JPXDecode'
        procset = 'ImageC'
        dict_obj['SMaskInData'] = 1
    elif im.mode == 'CMYK':
        filter = 'DCTDecode'
        dict_obj['ColorSpace'] = PdfParser.PdfName('DeviceCMYK')
        procset = 'ImageC'
        decode = [1, 0, 1, 0, 1, 0, 1, 0]
    else:
        msg = f'cannot save mode {im.mode}'
        raise ValueError(msg)
    op = io.BytesIO()
    if filter == 'ASCIIHexDecode':
        ImageFile._save(im, op, [('hex', (0, 0) + im.size, 0, im.mode)])
    elif filter == 'CCITTFaxDecode':
        im.save(op, 'TIFF', compression='group4', strip_size=math.ceil(width / 8) * height)
    elif filter == 'DCTDecode':
        Image.SAVE['JPEG'](im, op, filename)
    elif filter == 'JPXDecode':
        del dict_obj['BitsPerComponent']
        Image.SAVE['JPEG2000'](im, op, filename)
    else:
        msg = f'unsupported PDF filter ({filter})'
        raise ValueError(msg)
    stream = op.getvalue()
    if filter == 'CCITTFaxDecode':
        stream = stream[8:]
        filter = PdfParser.PdfArray([PdfParser.PdfName(filter)])
    else:
        filter = PdfParser.PdfName(filter)
    image_ref = image_refs.pop(0)
    existing_pdf.write_obj(image_ref, stream=stream, Type=PdfParser.PdfName('XObject'), Subtype=PdfParser.PdfName('Image'), Width=width, Height=height, Filter=filter, Decode=decode, DecodeParms=params, **dict_obj)
    return (image_ref, procset)