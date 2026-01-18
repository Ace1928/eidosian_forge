from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def try_to_guess(thing, valtype):
    if '.' not in thing and '(' not in thing:
        text = '{}.{}'.format(valtype, thing)
        ret = make_good_value(text, valtype)
        if ret is not None:
            return ret
    typewords = valtype.split('.')
    valwords = thing.split('.')
    braceless = valwords[0]
    if '(' in braceless:
        braceless = braceless[:braceless.index('(')]
    for idx, w in enumerate(typewords):
        if w == braceless:
            text = '.'.join(typewords[:idx] + valwords)
            ret = make_good_value(text, valtype)
            if ret is not None:
                return ret
    return None