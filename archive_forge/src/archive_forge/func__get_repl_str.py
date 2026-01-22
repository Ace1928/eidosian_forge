import warnings
from typing import Dict, Optional, Sequence
def _get_repl_str(char: str) -> Optional[str]:
    codepoint = ord(char)
    if codepoint < 128:
        return str(char)
    if codepoint > 983039:
        return None
    if 55296 <= codepoint <= 57343:
        warnings.warn('Surrogate character %r will be ignored. You might be using a narrow Python build.' % (char,), RuntimeWarning, 2)
    section = codepoint >> 8
    position = codepoint % 256
    try:
        table = Cache[section]
    except KeyError:
        try:
            mod = __import__('unidecode.x%03x' % section, globals(), locals(), ['data'])
        except ImportError:
            Cache[section] = None
            return None
        Cache[section] = table = mod.data
    if table and len(table) > position:
        return table[position]
    else:
        return None