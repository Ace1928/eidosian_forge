from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class CharStrings(object):
    """The ``CharStrings`` in the font represent the instructions for drawing
    each glyph. This object presents a dictionary interface to the font's
    CharStrings, indexed by glyph name:

    .. code:: python

            tt["CFF "].cff[0].CharStrings["a"]
            # <T2CharString (bytecode) at 103451e90>

    See :class:`fontTools.misc.psCharStrings.T1CharString` and
    :class:`fontTools.misc.psCharStrings.T2CharString` for how to decompile,
    compile and interpret the glyph drawing instructions in the returned objects.

    """

    def __init__(self, file, charset, globalSubrs, private, fdSelect, fdArray, isCFF2=None, varStore=None):
        self.globalSubrs = globalSubrs
        self.varStore = varStore
        if file is not None:
            self.charStringsIndex = SubrsIndex(file, globalSubrs, private, fdSelect, fdArray, isCFF2=isCFF2)
            self.charStrings = charStrings = {}
            for i in range(len(charset)):
                charStrings[charset[i]] = i
            self.charStringsAreIndexed = 1
        else:
            self.charStrings = {}
            self.charStringsAreIndexed = 0
            self.private = private
            if fdSelect is not None:
                self.fdSelect = fdSelect
            if fdArray is not None:
                self.fdArray = fdArray

    def keys(self):
        return list(self.charStrings.keys())

    def values(self):
        if self.charStringsAreIndexed:
            return self.charStringsIndex
        else:
            return list(self.charStrings.values())

    def has_key(self, name):
        return name in self.charStrings
    __contains__ = has_key

    def __len__(self):
        return len(self.charStrings)

    def __getitem__(self, name):
        charString = self.charStrings[name]
        if self.charStringsAreIndexed:
            charString = self.charStringsIndex[charString]
        return charString

    def __setitem__(self, name, charString):
        if self.charStringsAreIndexed:
            index = self.charStrings[name]
            self.charStringsIndex[index] = charString
        else:
            self.charStrings[name] = charString

    def getItemAndSelector(self, name):
        if self.charStringsAreIndexed:
            index = self.charStrings[name]
            return self.charStringsIndex.getItemAndSelector(index)
        else:
            if hasattr(self, 'fdArray'):
                if hasattr(self, 'fdSelect'):
                    sel = self.charStrings[name].fdSelectIndex
                else:
                    sel = 0
            else:
                sel = None
            return (self.charStrings[name], sel)

    def toXML(self, xmlWriter):
        names = sorted(self.keys())
        for name in names:
            charStr, fdSelectIndex = self.getItemAndSelector(name)
            if charStr.needsDecompilation():
                raw = [('raw', 1)]
            else:
                raw = []
            if fdSelectIndex is None:
                xmlWriter.begintag('CharString', [('name', name)] + raw)
            else:
                xmlWriter.begintag('CharString', [('name', name), ('fdSelectIndex', fdSelectIndex)] + raw)
            xmlWriter.newline()
            charStr.toXML(xmlWriter)
            xmlWriter.endtag('CharString')
            xmlWriter.newline()

    def fromXML(self, name, attrs, content):
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            if name != 'CharString':
                continue
            fdID = -1
            if hasattr(self, 'fdArray'):
                try:
                    fdID = safeEval(attrs['fdSelectIndex'])
                except KeyError:
                    fdID = 0
                private = self.fdArray[fdID].Private
            else:
                private = self.private
            glyphName = attrs['name']
            charStringClass = psCharStrings.T2CharString
            charString = charStringClass(private=private, globalSubrs=self.globalSubrs)
            charString.fromXML(name, attrs, content)
            if fdID >= 0:
                charString.fdSelectIndex = fdID
            self[glyphName] = charString