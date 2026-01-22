from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _coerce_args(*args):
    str_input = isinstance(args[0], str)
    for arg in args[1:]:
        if arg and isinstance(arg, str) != str_input:
            raise TypeError('Cannot mix str and non-str arguments')
    if str_input:
        return args + (_noop,)
    return _decode_args(args) + (_encode_result,)