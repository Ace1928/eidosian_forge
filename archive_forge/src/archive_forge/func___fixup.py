from __future__ import annotations
import builtins
from . import Image, _imagingmath
def __fixup(self, im1):
    if isinstance(im1, _Operand):
        if im1.im.mode in ('1', 'L'):
            return im1.im.convert('I')
        elif im1.im.mode in ('I', 'F'):
            return im1.im
        else:
            msg = f'unsupported mode: {im1.im.mode}'
            raise ValueError(msg)
    elif isinstance(im1, (int, float)) and self.im.mode in ('1', 'L', 'I'):
        return Image.new('I', self.im.size, im1)
    else:
        return Image.new('F', self.im.size, im1)