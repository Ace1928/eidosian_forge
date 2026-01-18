from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def zipfndict(*args, **kw):
    default = kw.get('default', nop)
    d = {}
    for key in unionlist(*(fndict.keys() for fndict in args)):
        d[key] = tuple((x.get(key, default) for x in args))
    return d