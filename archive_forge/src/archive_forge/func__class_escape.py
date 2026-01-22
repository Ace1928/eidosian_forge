from ._constants import *
def _class_escape(source, escape):
    code = ESCAPES.get(escape)
    if code:
        return code
    code = CATEGORIES.get(escape)
    if code and code[0] is IN:
        return code
    try:
        c = escape[1:2]
        if c == 'x':
            escape += source.getwhile(2, HEXDIGITS)
            if len(escape) != 4:
                raise source.error('incomplete escape %s' % escape, len(escape))
            return (LITERAL, int(escape[2:], 16))
        elif c == 'u' and source.istext:
            escape += source.getwhile(4, HEXDIGITS)
            if len(escape) != 6:
                raise source.error('incomplete escape %s' % escape, len(escape))
            return (LITERAL, int(escape[2:], 16))
        elif c == 'U' and source.istext:
            escape += source.getwhile(8, HEXDIGITS)
            if len(escape) != 10:
                raise source.error('incomplete escape %s' % escape, len(escape))
            c = int(escape[2:], 16)
            chr(c)
            return (LITERAL, c)
        elif c == 'N' and source.istext:
            import unicodedata
            if not source.match('{'):
                raise source.error('missing {')
            charname = source.getuntil('}', 'character name')
            try:
                c = ord(unicodedata.lookup(charname))
            except (KeyError, TypeError):
                raise source.error('undefined character name %r' % charname, len(charname) + len('\\N{}')) from None
            return (LITERAL, c)
        elif c in OCTDIGITS:
            escape += source.getwhile(2, OCTDIGITS)
            c = int(escape[1:], 8)
            if c > 255:
                raise source.error('octal escape value %s outside of range 0-0o377' % escape, len(escape))
            return (LITERAL, c)
        elif c in DIGITS:
            raise ValueError
        if len(escape) == 2:
            if c in ASCIILETTERS:
                raise source.error('bad escape %s' % escape, len(escape))
            return (LITERAL, ord(escape[1]))
    except ValueError:
        pass
    raise source.error('bad escape %s' % escape, len(escape))