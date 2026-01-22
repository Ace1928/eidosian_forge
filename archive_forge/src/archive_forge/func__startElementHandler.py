from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging
def _startElementHandler(self, name, attrs):
    if self.stackSize == 1 and self.contentOnly:
        self.contentStack.append([])
        self.stackSize = 2
        return
    stackSize = self.stackSize
    self.stackSize = stackSize + 1
    subFile = attrs.get('src')
    if subFile is not None:
        if hasattr(self.file, 'name'):
            dirname = os.path.dirname(self.file.name)
        else:
            dirname = os.getcwd()
        subFile = os.path.join(dirname, subFile)
    if not stackSize:
        if name != 'ttFont':
            raise TTXParseError('illegal root tag: %s' % name)
        if self.ttFont.reader is None and (not self.ttFont.tables):
            sfntVersion = attrs.get('sfntVersion')
            if sfntVersion is not None:
                if len(sfntVersion) != 4:
                    sfntVersion = safeEval('"' + sfntVersion + '"')
                self.ttFont.sfntVersion = sfntVersion
        self.contentStack.append([])
    elif stackSize == 1:
        if subFile is not None:
            subReader = XMLReader(subFile, self.ttFont, self.progress)
            subReader.read()
            self.contentStack.append([])
            return
        tag = ttLib.xmlToTag(name)
        msg = "Parsing '%s' table..." % tag
        if self.progress:
            self.progress.setLabel(msg)
        log.info(msg)
        if tag == 'GlyphOrder':
            tableClass = ttLib.GlyphOrder
        elif 'ERROR' in attrs or ('raw' in attrs and safeEval(attrs['raw'])):
            tableClass = DefaultTable
        else:
            tableClass = ttLib.getTableClass(tag)
            if tableClass is None:
                tableClass = DefaultTable
        if tag == 'loca' and tag in self.ttFont:
            self.currentTable = self.ttFont[tag]
        else:
            self.currentTable = tableClass(tag)
            self.ttFont[tag] = self.currentTable
        self.contentStack.append([])
    elif stackSize == 2 and subFile is not None:
        subReader = XMLReader(subFile, self.ttFont, self.progress, contentOnly=True)
        subReader.read()
        self.contentStack.append([])
        self.root = subReader.root
    elif stackSize == 2:
        self.contentStack.append([])
        self.root = (name, attrs, self.contentStack[-1])
    else:
        l = []
        self.contentStack[-1].append((name, attrs, l))
        self.contentStack.append(l)