from fontTools.misc.visitor import Visitor
from fontTools.ttLib import TTFont
def visitAttr(self, obj, attr, value, *args, **kwargs):
    if isinstance(value, TTFont):
        return False
    super().visitAttr(obj, attr, value, *args, **kwargs)