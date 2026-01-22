from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _parse_signature(self, args, kw):
    values = {}
    sig_args, var_args, var_kw, defaults = self._func_signature
    extra_kw = {}
    for name, value in kw.items():
        if not var_kw and name not in sig_args:
            raise TypeError('Unexpected argument %s' % name)
        if name in sig_args:
            values[sig_args] = value
        else:
            extra_kw[name] = value
    args = list(args)
    sig_args = list(sig_args)
    while args:
        while sig_args and sig_args[0] in values:
            sig_args.pop(0)
        if sig_args:
            name = sig_args.pop(0)
            values[name] = args.pop(0)
        elif var_args:
            values[var_args] = tuple(args)
            break
        else:
            raise TypeError('Extra position arguments: %s' % ', '.join([repr(v) for v in args]))
    for name, value_expr in defaults.items():
        if name not in values:
            values[name] = self._template._eval(value_expr, self._ns, self._pos)
    for name in sig_args:
        if name not in values:
            raise TypeError('Missing argument: %s' % name)
    if var_kw:
        values[var_kw] = extra_kw
    return values