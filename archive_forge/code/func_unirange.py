import re
import sys
def unirange(a, b):
    """Returns a regular expression string to match the given non-BMP range."""
    if b < a:
        raise ValueError('Bad character range')
    if a < 65536 or b < 65536:
        raise ValueError('unirange is only defined for non-BMP ranges')
    if sys.maxunicode > 65535:
        return u'[%s-%s]' % (unichr(a), unichr(b))
    else:
        ah, al = _surrogatepair(a)
        bh, bl = _surrogatepair(b)
        if ah == bh:
            return u'(?:%s[%s-%s])' % (unichr(ah), unichr(al), unichr(bl))
        else:
            buf = []
            buf.append(u'%s[%s-%s]' % (unichr(ah), unichr(al), ah == bh and unichr(bl) or unichr(57343)))
            if ah - bh > 1:
                buf.append(u'[%s-%s][%s-%s]' % unichr(ah + 1), unichr(bh - 1), unichr(56320), unichr(57343))
            if ah != bh:
                buf.append(u'%s[%s-%s]' % (unichr(bh), unichr(56320), unichr(bl)))
            return u'(?:' + u'|'.join(buf) + u')'