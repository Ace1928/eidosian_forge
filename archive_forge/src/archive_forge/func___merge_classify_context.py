from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
@add_method(otTables.ContextSubst, otTables.ChainContextSubst, otTables.ContextPos, otTables.ChainContextPos)
def __merge_classify_context(self):

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
            else:
                Chain = ''
            ChainTyp = Chain + Typ
            self.Typ = Typ
            self.Type = Type
            self.Chain = Chain
            self.ChainTyp = ChainTyp
            self.LookupRecord = Type + 'LookupRecord'
            if Format == 1:
                self.Rule = ChainTyp + 'Rule'
                self.RuleSet = ChainTyp + 'RuleSet'
            elif Format == 2:
                self.Rule = ChainTyp + 'ClassRule'
                self.RuleSet = ChainTyp + 'ClassSet'
    if self.Format not in [1, 2, 3]:
        return None
    if not hasattr(self.__class__, '_merge__ContextHelpers'):
        self.__class__._merge__ContextHelpers = {}
    if self.Format not in self.__class__._merge__ContextHelpers:
        helper = ContextHelper(self.__class__, self.Format)
        self.__class__._merge__ContextHelpers[self.Format] = helper
    return self.__class__._merge__ContextHelpers[self.Format]