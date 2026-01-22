from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import struct
class VOriginRecord(object):

    def __init__(self, name=None, vOrigin=None):
        self.glyphName = name
        self.vOrigin = vOrigin

    def toXML(self, writer, ttFont):
        writer.begintag('VOriginRecord')
        writer.newline()
        writer.simpletag('glyphName', value=self.glyphName)
        writer.newline()
        writer.simpletag('vOrigin', value=self.vOrigin)
        writer.newline()
        writer.endtag('VOriginRecord')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs['value']
        if name == 'glyphName':
            setattr(self, name, value)
        else:
            setattr(self, name, safeEval(value))