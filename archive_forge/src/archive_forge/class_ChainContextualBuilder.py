from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
class ChainContextualBuilder(LookupBuilder):

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def rulesets(self):
        ruleset = [ChainContextualRuleset()]
        for rule in self.rules:
            if rule.is_subtable_break:
                ruleset.append(ChainContextualRuleset())
                continue
            ruleset[-1].addRule(rule)
        return [x for x in ruleset if len(x.rules) > 0]

    def getCompiledSize_(self, subtables):
        if not subtables:
            return 0
        table = self.buildLookup_(copy.deepcopy(subtables))
        w = OTTableWriter()
        table.compile(w, self.font)
        size = len(w.getAllData())
        return size

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual positioning lookup.
        """
        subtables = []
        rulesets = self.rulesets()
        chaining = any((ruleset.hasPrefixOrSuffix for ruleset in rulesets))
        write_gpos7 = self.font.cfg.get('fontTools.otlLib.builder:WRITE_GPOS7')
        if not write_gpos7 and self.subtable_type == 'Pos':
            chaining = True
        for ruleset in rulesets:
            candidates = [None, None, None, []]
            for rule in ruleset.rules:
                candidates[3].append(self.buildFormat3Subtable(rule, chaining))
            classdefs = ruleset.format2ClassDefs()
            if classdefs:
                candidates[2] = [self.buildFormat2Subtable(ruleset, classdefs, chaining)]
            if not ruleset.hasAnyGlyphClasses:
                candidates[1] = [self.buildFormat1Subtable(ruleset, chaining)]
            candidates_by_size = []
            for i in [1, 2, 3]:
                if candidates[i]:
                    try:
                        size = self.getCompiledSize_(candidates[i])
                    except OTLOffsetOverflowError as e:
                        log.warning('Contextual format %i at %s overflowed (%s)' % (i, str(self.location), e))
                    else:
                        candidates_by_size.append((size, candidates[i]))
            if not candidates_by_size:
                raise OpenTypeLibError('All candidates overflowed', self.location)
            _min_size, winner = min(candidates_by_size, key=lambda x: x[0])
            subtables.extend(winner)
        return self.buildLookup_(subtables)

    def buildFormat1Subtable(self, ruleset, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 1
        st.populateDefaults()
        coverage = set()
        rulesetsByFirstGlyph = {}
        ruleAttr = self.ruleAttr_(format=1, chaining=chaining)
        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=1, chaining=chaining)
            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                ruleAsSubtable.Backtrack = [list(x)[0] for x in reversed(rule.prefix)]
                ruleAsSubtable.LookAhead = [list(x)[0] for x in rule.suffix]
                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)
            ruleAsSubtable.Input = [list(x)[0] for x in rule.glyphs[1:]]
            self.buildLookupList(rule, ruleAsSubtable)
            firstGlyph = list(rule.glyphs[0])[0]
            if firstGlyph not in rulesetsByFirstGlyph:
                coverage.add(firstGlyph)
                rulesetsByFirstGlyph[firstGlyph] = []
            rulesetsByFirstGlyph[firstGlyph].append(ruleAsSubtable)
        st.Coverage = buildCoverage(coverage, self.glyphMap)
        ruleSets = []
        for g in st.Coverage.glyphs:
            ruleSet = self.newRuleSet_(format=1, chaining=chaining)
            setattr(ruleSet, ruleAttr, rulesetsByFirstGlyph[g])
            setattr(ruleSet, f'{ruleAttr}Count', len(rulesetsByFirstGlyph[g]))
            ruleSets.append(ruleSet)
        setattr(st, self.ruleSetAttr_(format=1, chaining=chaining), ruleSets)
        setattr(st, self.ruleSetAttr_(format=1, chaining=chaining) + 'Count', len(ruleSets))
        return st

    def buildFormat2Subtable(self, ruleset, classdefs, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 2
        st.populateDefaults()
        if chaining:
            st.BacktrackClassDef, st.InputClassDef, st.LookAheadClassDef = [c.build() for c in classdefs]
        else:
            st.ClassDef = classdefs[1].build()
        inClasses = classdefs[1].classes()
        classSets = []
        for _ in inClasses:
            classSet = self.newRuleSet_(format=2, chaining=chaining)
            classSets.append(classSet)
        coverage = set()
        classRuleAttr = self.ruleAttr_(format=2, chaining=chaining)
        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=2, chaining=chaining)
            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                ruleAsSubtable.Backtrack = [st.BacktrackClassDef.classDefs[list(x)[0]] for x in reversed(rule.prefix)]
                ruleAsSubtable.LookAhead = [st.LookAheadClassDef.classDefs[list(x)[0]] for x in rule.suffix]
                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
                ruleAsSubtable.Input = [st.InputClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]]
                setForThisRule = classSets[st.InputClassDef.classDefs[list(rule.glyphs[0])[0]]]
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)
                ruleAsSubtable.Class = [st.ClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]]
                setForThisRule = classSets[st.ClassDef.classDefs[list(rule.glyphs[0])[0]]]
            self.buildLookupList(rule, ruleAsSubtable)
            coverage |= set(rule.glyphs[0])
            getattr(setForThisRule, classRuleAttr).append(ruleAsSubtable)
            setattr(setForThisRule, f'{classRuleAttr}Count', getattr(setForThisRule, f'{classRuleAttr}Count') + 1)
        setattr(st, self.ruleSetAttr_(format=2, chaining=chaining), classSets)
        setattr(st, self.ruleSetAttr_(format=2, chaining=chaining) + 'Count', len(classSets))
        st.Coverage = buildCoverage(coverage, self.glyphMap)
        return st

    def buildFormat3Subtable(self, rule, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 3
        if chaining:
            self.setBacktrackCoverage_(rule.prefix, st)
            self.setLookAheadCoverage_(rule.suffix, st)
            self.setInputCoverage_(rule.glyphs, st)
        else:
            self.setCoverage_(rule.glyphs, st)
        self.buildLookupList(rule, st)
        return st

    def buildLookupList(self, rule, st):
        for sequenceIndex, lookupList in enumerate(rule.lookups):
            if lookupList is not None:
                if not isinstance(lookupList, list):
                    lookupList = [lookupList]
                for l in lookupList:
                    if l.lookup_index is None:
                        if isinstance(self, ChainContextPosBuilder):
                            other = 'substitution'
                        else:
                            other = 'positioning'
                        raise OpenTypeLibError(f'Missing index of the specified lookup, might be a {other} lookup', self.location)
                    rec = self.newLookupRecord_(st)
                    rec.SequenceIndex = sequenceIndex
                    rec.LookupListIndex = l.lookup_index

    def add_subtable_break(self, location):
        self.rules.append(ChainContextualRule(self.SUBTABLE_BREAK_, self.SUBTABLE_BREAK_, self.SUBTABLE_BREAK_, [self.SUBTABLE_BREAK_]))

    def newSubtable_(self, chaining=True):
        subtablename = f'Context{self.subtable_type}'
        if chaining:
            subtablename = 'Chain' + subtablename
        st = getattr(ot, subtablename)()
        setattr(st, f'{self.subtable_type}Count', 0)
        setattr(st, f'{self.subtable_type}LookupRecord', [])
        return st

    def ruleSetAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = 'Rule'
        elif format == 2:
            formatType = 'Class'
        else:
            raise AssertionError(formatType)
        subtablename = f'{self.subtable_type[0:3]}{formatType}Set'
        if chaining:
            subtablename = 'Chain' + subtablename
        return subtablename

    def ruleAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = ''
        elif format == 2:
            formatType = 'Class'
        else:
            raise AssertionError(formatType)
        subtablename = f'{self.subtable_type[0:3]}{formatType}Rule'
        if chaining:
            subtablename = 'Chain' + subtablename
        return subtablename

    def newRuleSet_(self, format=1, chaining=True):
        st = getattr(ot, self.ruleSetAttr_(format, chaining))()
        st.populateDefaults()
        return st

    def newRule_(self, format=1, chaining=True):
        st = getattr(ot, self.ruleAttr_(format, chaining))()
        st.populateDefaults()
        return st

    def attachSubtableWithCount_(self, st, subtable_name, count_name, existing=None, index=None, chaining=False):
        if chaining:
            subtable_name = 'Chain' + subtable_name
            count_name = 'Chain' + count_name
        if not hasattr(st, count_name):
            setattr(st, count_name, 0)
            setattr(st, subtable_name, [])
        if existing:
            new_subtable = existing
        else:
            new_subtable = getattr(ot, subtable_name)()
        setattr(st, count_name, getattr(st, count_name) + 1)
        if index:
            getattr(st, subtable_name).insert(index, new_subtable)
        else:
            getattr(st, subtable_name).append(new_subtable)
        return new_subtable

    def newLookupRecord_(self, st):
        return self.attachSubtableWithCount_(st, f'{self.subtable_type}LookupRecord', f'{self.subtable_type}Count', chaining=False)