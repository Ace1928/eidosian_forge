from fontTools.misc.textTools import safeEval
from . import DefaultTable
class LayerRecord(object):

    def __init__(self, name=None, colorID=None):
        self.name = name
        self.colorID = colorID

    def toXML(self, writer, ttFont):
        writer.simpletag('layer', name=self.name, colorID=self.colorID)
        writer.newline()

    def fromXML(self, eltname, attrs, content, ttFont):
        for name, value in attrs.items():
            if name == 'name':
                setattr(self, name, value)
            else:
                setattr(self, name, safeEval(value))