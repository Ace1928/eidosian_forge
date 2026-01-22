import copy
import re
from collections import namedtuple
def _normalizeOpts(options):
    convertedOpts = copy.copy(options)
    if isinstance(convertedOpts, dict):
        option_keys = list(convertedOpts.keys())
        for key in option_keys:
            if '-' in key:
                del convertedOpts[key]
                convertedOpts[key.replace('-', '_')] = options[key]
    else:
        option_keys = list(getattr(convertedOpts, '__dict__', {}))
        for key in option_keys:
            if '-' in key:
                delattr(convertedOpts, key)
                setattr(convertedOpts, key.replace('-', '_'), getattr(options, key, None))
    return convertedOpts