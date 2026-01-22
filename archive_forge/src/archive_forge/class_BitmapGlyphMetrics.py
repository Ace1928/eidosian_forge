from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import logging
class BitmapGlyphMetrics(object):

    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__)
        writer.newline()
        for metricName in sstruct.getformat(self.__class__.binaryFormat)[1]:
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        metricNames = set(sstruct.getformat(self.__class__.binaryFormat)[1])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name in metricNames:
                vars(self)[name] = safeEval(attrs['value'])
            else:
                log.warning("unknown name '%s' being ignored in %s.", name, self.__class__.__name__)