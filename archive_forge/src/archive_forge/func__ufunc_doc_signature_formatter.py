import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _ufunc_doc_signature_formatter(ufunc):
    """
    Builds a signature string which resembles PEP 457

    This is used to construct the first line of the docstring
    """
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join((f'x{i + 1}' for i in range(ufunc.nin)))
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(positional=', '.join(('out{}'.format(i + 1) for i in range(ufunc.nout))), default=repr((None,) * ufunc.nout))
    kwargs = ", casting='same_kind', order='K', dtype=None, subok=True"
    if ufunc.signature is None:
        kwargs = f', where=True{kwargs}[, signature, extobj]'
    else:
        kwargs += '[, signature, extobj, axes, axis]'
    return '{name}({in_args}{out_args}, *{kwargs})'.format(name=ufunc.__name__, in_args=in_args, out_args=out_args, kwargs=kwargs)