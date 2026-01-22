import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
class ClipList(getFormatSwitchingBaseTableClass('uint8')):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'clips'):
            self.clips = {}

    def postRead(self, rawTable, font):
        clips = {}
        glyphOrder = font.getGlyphOrder()
        for i, rec in enumerate(rawTable['ClipRecord']):
            if rec.StartGlyphID > rec.EndGlyphID:
                log.warning('invalid ClipRecord[%i].StartGlyphID (%i) > EndGlyphID (%i); skipped', i, rec.StartGlyphID, rec.EndGlyphID)
                continue
            redefinedGlyphs = []
            missingGlyphs = []
            for glyphID in range(rec.StartGlyphID, rec.EndGlyphID + 1):
                try:
                    glyph = glyphOrder[glyphID]
                except IndexError:
                    missingGlyphs.append(glyphID)
                    continue
                if glyph not in clips:
                    clips[glyph] = copy.copy(rec.ClipBox)
                else:
                    redefinedGlyphs.append(glyphID)
            if redefinedGlyphs:
                log.warning('ClipRecord[%i] overlaps previous records; ignoring redefined clip boxes for the following glyph ID range: [%i-%i]', i, min(redefinedGlyphs), max(redefinedGlyphs))
            if missingGlyphs:
                log.warning('ClipRecord[%i] range references missing glyph IDs: [%i-%i]', i, min(missingGlyphs), max(missingGlyphs))
        self.clips = clips

    def groups(self):
        glyphsByClip = defaultdict(list)
        uniqueClips = {}
        for glyphName, clipBox in self.clips.items():
            key = clipBox.as_tuple()
            glyphsByClip[key].append(glyphName)
            if key not in uniqueClips:
                uniqueClips[key] = clipBox
        return {frozenset(glyphs): uniqueClips[key] for key, glyphs in glyphsByClip.items()}

    def preWrite(self, font):
        if not hasattr(self, 'clips'):
            self.clips = {}
        clipBoxRanges = {}
        glyphMap = font.getReverseGlyphMap()
        for glyphs, clipBox in self.groups().items():
            glyphIDs = sorted((glyphMap[glyphName] for glyphName in glyphs if glyphName in glyphMap))
            if not glyphIDs:
                continue
            last = glyphIDs[0]
            ranges = [[last]]
            for glyphID in glyphIDs[1:]:
                if glyphID != last + 1:
                    ranges[-1].append(last)
                    ranges.append([glyphID])
                last = glyphID
            ranges[-1].append(last)
            for start, end in ranges:
                assert (start, end) not in clipBoxRanges
                clipBoxRanges[start, end] = clipBox
        clipRecords = []
        for (start, end), clipBox in sorted(clipBoxRanges.items()):
            record = ClipRecord()
            record.StartGlyphID = start
            record.EndGlyphID = end
            record.ClipBox = clipBox
            clipRecords.append(record)
        rawTable = {'ClipCount': len(clipRecords), 'ClipRecord': clipRecords}
        return rawTable

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        tableName = name if name else self.__class__.__name__
        if attrs is None:
            attrs = []
        if hasattr(self, 'Format'):
            attrs.append(('Format', self.Format))
        xmlWriter.begintag(tableName, attrs)
        xmlWriter.newline()
        for glyphs, clipBox in sorted(self.groups().items(), key=lambda item: min(item[0])):
            xmlWriter.begintag('Clip')
            xmlWriter.newline()
            for glyphName in sorted(glyphs):
                xmlWriter.simpletag('Glyph', value=glyphName)
                xmlWriter.newline()
            xmlWriter.begintag('ClipBox', [('Format', clipBox.Format)])
            xmlWriter.newline()
            clipBox.toXML2(xmlWriter, font)
            xmlWriter.endtag('ClipBox')
            xmlWriter.newline()
            xmlWriter.endtag('Clip')
            xmlWriter.newline()
        xmlWriter.endtag(tableName)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        clips = getattr(self, 'clips', None)
        if clips is None:
            self.clips = clips = {}
        assert name == 'Clip'
        glyphs = []
        clipBox = None
        for elem in content:
            if not isinstance(elem, tuple):
                continue
            name, attrs, content = elem
            if name == 'Glyph':
                glyphs.append(attrs['value'])
            elif name == 'ClipBox':
                clipBox = ClipBox()
                clipBox.Format = safeEval(attrs['Format'])
                for elem in content:
                    if not isinstance(elem, tuple):
                        continue
                    name, attrs, content = elem
                    clipBox.fromXML(name, attrs, content, font)
        if clipBox:
            for glyphName in glyphs:
                clips[glyphName] = clipBox