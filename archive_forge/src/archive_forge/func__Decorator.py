from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def _Decorator(fn):
    parse_fns = GetParseFns(fn)
    parse_fns['positional'] = positional
    parse_fns['named'].update(named)
    _SetMetadata(fn, FIRE_PARSE_FNS, parse_fns)
    return fn