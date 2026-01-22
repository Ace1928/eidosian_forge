from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
def addProxyAttribute(src, name, validate=None, desc=None, initial=None, dst=None):
    """
    Add a proxy attribute 'name' to src with targets dst
    """
    assert hasattr(src, '_attrMap'), 'src object has no _attrMap'
    A, oA = _privateAttrMap(src, 1)
    if not isSeq(dst):
        dst = (dst,)
    D = []
    DV = []
    for d in dst:
        if isSeq(d):
            d, e = (d[0], d[1:])
        obj, attr = _findObjectAndAttr(src, d)
        if obj:
            dA = getattr(obj, '_attrMap', None)