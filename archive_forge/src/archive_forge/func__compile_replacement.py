import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def _compile_replacement(source, pattern, is_unicode):
    """Compiles a replacement template escape sequence."""
    ch = source.get()
    if ch in ALPHA:
        value = CHARACTER_ESCAPES.get(ch)
        if value:
            return (False, [ord(value)])
        if ch in HEX_ESCAPES and (ch == 'x' or is_unicode):
            return (False, [parse_repl_hex_escape(source, HEX_ESCAPES[ch], ch)])
        if ch == 'g':
            return (True, [compile_repl_group(source, pattern)])
        if ch == 'N' and is_unicode:
            value = parse_repl_named_char(source)
            if value is not None:
                return (False, [value])
        raise error('bad escape \\%s' % ch, source.string, source.pos)
    if isinstance(source.sep, bytes):
        octal_mask = 255
    else:
        octal_mask = 511
    if ch == '0':
        digits = ch
        while len(digits) < 3:
            saved_pos = source.pos
            ch = source.get()
            if ch not in OCT_DIGITS:
                source.pos = saved_pos
                break
            digits += ch
        return (False, [int(digits, 8) & octal_mask])
    if ch in DIGITS:
        digits = ch
        saved_pos = source.pos
        ch = source.get()
        if ch in DIGITS:
            digits += ch
            saved_pos = source.pos
            ch = source.get()
            if ch and is_octal(digits + ch):
                return (False, [int(digits + ch, 8) & octal_mask])
        source.pos = saved_pos
        return (True, [int(digits)])
    if ch == '\\':
        return (False, [ord('\\')])
    if not ch:
        raise error('bad escape (end of pattern)', source.string, source.pos)
    return (False, [ord('\\'), ord(ch)])