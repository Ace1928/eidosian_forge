from __future__ import absolute_import
import re
import sys
def _build_specials_replacer():
    subexps = []
    replacements = {}
    for special in _c_special:
        regexp = ''.join(['[%s]' % c.replace('\\', '\\\\') for c in special])
        subexps.append(regexp)
        replacements[special.encode('ASCII')] = _to_escape_sequence(special).encode('ASCII')
    sub = re.compile(('(%s)' % '|'.join(subexps)).encode('ASCII')).sub

    def replace_specials(m):
        return replacements[m.group(1)]

    def replace(s):
        return sub(replace_specials, s)
    return replace