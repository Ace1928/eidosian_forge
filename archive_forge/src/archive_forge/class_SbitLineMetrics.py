from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class SbitLineMetrics(object):

    def toXML(self, name, writer, ttFont):
        writer.begintag('sbitLineMetrics', [('direction', name)])
        writer.newline()
        for metricName in sstruct.getformat(sbitLineMetricsFormat)[1]:
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag('sbitLineMetrics')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        metricNames = set(sstruct.getformat(sbitLineMetricsFormat)[1])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name in metricNames:
                vars(self)[name] = safeEval(attrs['value'])