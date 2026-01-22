from __future__ import annotations
import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
def _bitmap(self, header=0, offset=0):
    """Read relevant info about the BMP"""
    read, seek = (self.fp.read, self.fp.seek)
    if header:
        seek(header)
    file_info = {'header_size': i32(read(4)), 'direction': -1}
    header_data = ImageFile._safe_read(self.fp, file_info['header_size'] - 4)
    if file_info['header_size'] == 12:
        file_info['width'] = i16(header_data, 0)
        file_info['height'] = i16(header_data, 2)
        file_info['planes'] = i16(header_data, 4)
        file_info['bits'] = i16(header_data, 6)
        file_info['compression'] = self.RAW
        file_info['palette_padding'] = 3
    elif file_info['header_size'] in (40, 64, 108, 124):
        file_info['y_flip'] = header_data[7] == 255
        file_info['direction'] = 1 if file_info['y_flip'] else -1
        file_info['width'] = i32(header_data, 0)
        file_info['height'] = i32(header_data, 4) if not file_info['y_flip'] else 2 ** 32 - i32(header_data, 4)
        file_info['planes'] = i16(header_data, 8)
        file_info['bits'] = i16(header_data, 10)
        file_info['compression'] = i32(header_data, 12)
        file_info['data_size'] = i32(header_data, 16)
        file_info['pixels_per_meter'] = (i32(header_data, 20), i32(header_data, 24))
        file_info['colors'] = i32(header_data, 28)
        file_info['palette_padding'] = 4
        self.info['dpi'] = tuple((x / 39.3701 for x in file_info['pixels_per_meter']))
        if file_info['compression'] == self.BITFIELDS:
            if len(header_data) >= 52:
                for idx, mask in enumerate(['r_mask', 'g_mask', 'b_mask', 'a_mask']):
                    file_info[mask] = i32(header_data, 36 + idx * 4)
            else:
                file_info['a_mask'] = 0
                for mask in ['r_mask', 'g_mask', 'b_mask']:
                    file_info[mask] = i32(read(4))
            file_info['rgb_mask'] = (file_info['r_mask'], file_info['g_mask'], file_info['b_mask'])
            file_info['rgba_mask'] = (file_info['r_mask'], file_info['g_mask'], file_info['b_mask'], file_info['a_mask'])
    else:
        msg = f'Unsupported BMP header type ({file_info['header_size']})'
        raise OSError(msg)
    self._size = (file_info['width'], file_info['height'])
    file_info['colors'] = file_info['colors'] if file_info.get('colors', 0) else 1 << file_info['bits']
    if offset == 14 + file_info['header_size'] and file_info['bits'] <= 8:
        offset += 4 * file_info['colors']
    self._mode, raw_mode = BIT2MODE.get(file_info['bits'], (None, None))
    if self.mode is None:
        msg = f'Unsupported BMP pixel depth ({file_info['bits']})'
        raise OSError(msg)
    decoder_name = 'raw'
    if file_info['compression'] == self.BITFIELDS:
        SUPPORTED = {32: [(16711680, 65280, 255, 0), (4278190080, 16711680, 65280, 0), (4278190080, 16711680, 65280, 255), (255, 65280, 16711680, 4278190080), (16711680, 65280, 255, 4278190080), (0, 0, 0, 0)], 24: [(16711680, 65280, 255)], 16: [(63488, 2016, 31), (31744, 992, 31)]}
        MASK_MODES = {(32, (16711680, 65280, 255, 0)): 'BGRX', (32, (4278190080, 16711680, 65280, 0)): 'XBGR', (32, (4278190080, 16711680, 65280, 255)): 'ABGR', (32, (255, 65280, 16711680, 4278190080)): 'RGBA', (32, (16711680, 65280, 255, 4278190080)): 'BGRA', (32, (0, 0, 0, 0)): 'BGRA', (24, (16711680, 65280, 255)): 'BGR', (16, (63488, 2016, 31)): 'BGR;16', (16, (31744, 992, 31)): 'BGR;15'}
        if file_info['bits'] in SUPPORTED:
            if file_info['bits'] == 32 and file_info['rgba_mask'] in SUPPORTED[file_info['bits']]:
                raw_mode = MASK_MODES[file_info['bits'], file_info['rgba_mask']]
                self._mode = 'RGBA' if 'A' in raw_mode else self.mode
            elif file_info['bits'] in (24, 16) and file_info['rgb_mask'] in SUPPORTED[file_info['bits']]:
                raw_mode = MASK_MODES[file_info['bits'], file_info['rgb_mask']]
            else:
                msg = 'Unsupported BMP bitfields layout'
                raise OSError(msg)
        else:
            msg = 'Unsupported BMP bitfields layout'
            raise OSError(msg)
    elif file_info['compression'] == self.RAW:
        if file_info['bits'] == 32 and header == 22:
            raw_mode, self._mode = ('BGRA', 'RGBA')
    elif file_info['compression'] in (self.RLE8, self.RLE4):
        decoder_name = 'bmp_rle'
    else:
        msg = f'Unsupported BMP compression ({file_info['compression']})'
        raise OSError(msg)
    if self.mode == 'P':
        if not 0 < file_info['colors'] <= 65536:
            msg = f'Unsupported BMP Palette size ({file_info['colors']})'
            raise OSError(msg)
        else:
            padding = file_info['palette_padding']
            palette = read(padding * file_info['colors'])
            grayscale = True
            indices = (0, 255) if file_info['colors'] == 2 else list(range(file_info['colors']))
            for ind, val in enumerate(indices):
                rgb = palette[ind * padding:ind * padding + 3]
                if rgb != o8(val) * 3:
                    grayscale = False
            if grayscale:
                self._mode = '1' if file_info['colors'] == 2 else 'L'
                raw_mode = self.mode
            else:
                self._mode = 'P'
                self.palette = ImagePalette.raw('BGRX' if padding == 4 else 'BGR', palette)
    self.info['compression'] = file_info['compression']
    args = [raw_mode]
    if decoder_name == 'bmp_rle':
        args.append(file_info['compression'] == self.RLE4)
    else:
        args.append(file_info['width'] * file_info['bits'] + 31 >> 3 & ~3)
    args.append(file_info['direction'])
    self.tile = [(decoder_name, (0, 0, file_info['width'], file_info['height']), offset or self.fp.tell(), tuple(args))]