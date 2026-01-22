import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _parse_gufunc_signature(signature):
    if not isinstance(signature, str):
        raise TypeError('Signature is not a string')
    if signature == '' or signature is None:
        raise ValueError('Signature cannot be empty')
    signature = signature.replace(' ', '')
    if not re.match(_SIGNATURE, signature):
        raise ValueError('Not a valid gufunc signature: {}'.format(signature))
    in_txt, out_txt = signature.split('->')
    ins = [tuple(x.split(',')) if x != '' else () for x in in_txt[1:-1].split('),(')]
    outs = [tuple(y.split(',')) if y != '' else () for y in out_txt[1:-1].split('),(')]
    if len(outs) > 1:
        raise ValueError('Currently more than 1 output is not supported')
    return (ins, outs)