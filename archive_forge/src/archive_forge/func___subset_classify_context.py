from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
@_add_method(otTables.ContextSubst, otTables.ChainContextSubst, otTables.ContextPos, otTables.ChainContextPos)
def __subset_classify_context(self):

    class ContextHelper(object):

        def __init__(self, klass, Format):
            if klass.__name__.endswith('Subst'):
                Typ = 'Sub'
                Type = 'Subst'
            else:
                Typ = 'Pos'
                Type = 'Pos'
            if klass.__name__.startswith('Chain'):
                Chain = 'Chain'
                InputIdx = 1
                DataLen = 3
            else:
                Chain = ''
                InputIdx = 0
                DataLen = 1
            ChainTyp = Chain + Typ
            self.Typ = Typ
            self.Type = Type
            self.Chain = Chain
            self.ChainTyp = ChainTyp
            self.InputIdx = InputIdx
            self.DataLen = DataLen
            self.LookupRecord = Type + 'LookupRecord'
            if Format == 1:
                Coverage = lambda r: r.Coverage
                ChainCoverage = lambda r: r.Coverage
                ContextData = lambda r: (None,)
                ChainContextData = lambda r: (None, None, None)
                SetContextData = None
                SetChainContextData = None
                RuleData = lambda r: (r.Input,)
                ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

                def SetRuleData(r, d):
                    r.Input, = d
                    r.GlyphCount, = (len(x) + 1 for x in d)

                def ChainSetRuleData(r, d):
                    r.Backtrack, r.Input, r.LookAhead = d
                    r.BacktrackGlyphCount, r.InputGlyphCount, r.LookAheadGlyphCount = (len(d[0]), len(d[1]) + 1, len(d[2]))
            elif Format == 2:
                Coverage = lambda r: r.Coverage
                ChainCoverage = lambda r: r.Coverage
                ContextData = lambda r: (r.ClassDef,)
                ChainContextData = lambda r: (r.BacktrackClassDef, r.InputClassDef, r.LookAheadClassDef)

                def SetContextData(r, d):
                    r.ClassDef, = d

                def SetChainContextData(r, d):
                    r.BacktrackClassDef, r.InputClassDef, r.LookAheadClassDef = d
                RuleData = lambda r: (r.Class,)
                ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

                def SetRuleData(r, d):
                    r.Class, = d
                    r.GlyphCount, = (len(x) + 1 for x in d)

                def ChainSetRuleData(r, d):
                    r.Backtrack, r.Input, r.LookAhead = d
                    r.BacktrackGlyphCount, r.InputGlyphCount, r.LookAheadGlyphCount = (len(d[0]), len(d[1]) + 1, len(d[2]))
            elif Format == 3:
                Coverage = lambda r: r.Coverage[0]
                ChainCoverage = lambda r: r.InputCoverage[0]
                ContextData = None
                ChainContextData = None
                SetContextData = None
                SetChainContextData = None
                RuleData = lambda r: r.Coverage
                ChainRuleData = lambda r: r.BacktrackCoverage + r.InputCoverage + r.LookAheadCoverage

                def SetRuleData(r, d):
                    r.Coverage, = d
                    r.GlyphCount, = (len(x) for x in d)

                def ChainSetRuleData(r, d):
                    r.BacktrackCoverage, r.InputCoverage, r.LookAheadCoverage = d
                    r.BacktrackGlyphCount, r.InputGlyphCount, r.LookAheadGlyphCount = (len(x) for x in d)
            else:
                assert 0, 'unknown format: %s' % Format
            if Chain:
                self.Coverage = ChainCoverage
                self.ContextData = ChainContextData
                self.SetContextData = SetChainContextData
                self.RuleData = ChainRuleData
                self.SetRuleData = ChainSetRuleData
            else:
                self.Coverage = Coverage
                self.ContextData = ContextData
                self.SetContextData = SetContextData
                self.RuleData = RuleData
                self.SetRuleData = SetRuleData
            if Format == 1:
                self.Rule = ChainTyp + 'Rule'
                self.RuleCount = ChainTyp + 'RuleCount'
                self.RuleSet = ChainTyp + 'RuleSet'
                self.RuleSetCount = ChainTyp + 'RuleSetCount'
                self.Intersect = lambda glyphs, c, r: [r] if r in glyphs else []
            elif Format == 2:
                self.Rule = ChainTyp + 'ClassRule'
                self.RuleCount = ChainTyp + 'ClassRuleCount'
                self.RuleSet = ChainTyp + 'ClassSet'
                self.RuleSetCount = ChainTyp + 'ClassSetCount'
                self.Intersect = lambda glyphs, c, r: c.intersect_class(glyphs, r) if c else set(glyphs) if r == 0 else set()
                self.ClassDef = 'InputClassDef' if Chain else 'ClassDef'
                self.ClassDefIndex = 1 if Chain else 0
                self.Input = 'Input' if Chain else 'Class'
            elif Format == 3:
                self.Input = 'InputCoverage' if Chain else 'Coverage'
    if self.Format not in [1, 2, 3]:
        return None
    if not hasattr(self.__class__, '_subset__ContextHelpers'):
        self.__class__._subset__ContextHelpers = {}
    if self.Format not in self.__class__._subset__ContextHelpers:
        helper = ContextHelper(self.__class__, self.Format)
        self.__class__._subset__ContextHelpers[self.Format] = helper
    return self.__class__._subset__ContextHelpers[self.Format]