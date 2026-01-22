import regex._regex_core as _regex_core
import regex._regex as _regex
from threading import RLock as _RLock
from locale import getpreferredencoding as _getpreferredencoding
from regex._regex_core import *
from regex._regex_core import (_ALL_VERSIONS, _ALL_ENCODINGS, _FirstSetError,
from regex._regex_core import (ALNUM as _ALNUM, Info as _Info, OP as _OP, Source
import copyreg as _copy_reg
def _compile_replacement_helper(pattern, template):
    """Compiles a replacement template."""
    key = (pattern.pattern, pattern.flags, template)
    compiled = _replacement_cache.get(key)
    if compiled is not None:
        return compiled
    if len(_replacement_cache) >= _MAXREPCACHE:
        _replacement_cache.clear()
    is_unicode = isinstance(template, str)
    source = _Source(template)
    if is_unicode:

        def make_string(char_codes):
            return ''.join((chr(c) for c in char_codes))
    else:

        def make_string(char_codes):
            return bytes(char_codes)
    compiled = []
    literal = []
    while True:
        ch = source.get()
        if not ch:
            break
        if ch == '\\':
            is_group, items = _compile_replacement(source, pattern, is_unicode)
            if is_group:
                if literal:
                    compiled.append(make_string(literal))
                    literal = []
                compiled.extend(items)
            else:
                literal.extend(items)
        else:
            literal.append(ord(ch))
    if literal:
        compiled.append(make_string(literal))
    _replacement_cache[key] = compiled
    return compiled