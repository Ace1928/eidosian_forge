import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
class NltkDrtBoxerDrsInterpreter:

    def __init__(self, occur_index=False):
        self._occur_index = occur_index

    def interpret(self, ex):
        """
        :param ex: ``AbstractBoxerDrs``
        :return: ``DrtExpression``
        """
        if isinstance(ex, BoxerDrs):
            drs = DRS([Variable(r) for r in ex.refs], list(map(self.interpret, ex.conds)))
            if ex.consequent is not None:
                drs.consequent = self.interpret(ex.consequent)
            return drs
        elif isinstance(ex, BoxerNot):
            return DrtNegatedExpression(self.interpret(ex.drs))
        elif isinstance(ex, BoxerPred):
            pred = self._add_occur_indexing(f'{ex.pos}_{ex.name}', ex)
            return self._make_atom(pred, ex.var)
        elif isinstance(ex, BoxerNamed):
            pred = self._add_occur_indexing(f'ne_{ex.type}_{ex.name}', ex)
            return self._make_atom(pred, ex.var)
        elif isinstance(ex, BoxerRel):
            pred = self._add_occur_indexing('%s' % ex.rel, ex)
            return self._make_atom(pred, ex.var1, ex.var2)
        elif isinstance(ex, BoxerProp):
            return DrtProposition(Variable(ex.var), self.interpret(ex.drs))
        elif isinstance(ex, BoxerEq):
            return DrtEqualityExpression(DrtVariableExpression(Variable(ex.var1)), DrtVariableExpression(Variable(ex.var2)))
        elif isinstance(ex, BoxerCard):
            pred = self._add_occur_indexing(f'card_{ex.type}_{ex.value}', ex)
            return self._make_atom(pred, ex.var)
        elif isinstance(ex, BoxerOr):
            return DrtOrExpression(self.interpret(ex.drs1), self.interpret(ex.drs2))
        elif isinstance(ex, BoxerWhq):
            drs1 = self.interpret(ex.drs1)
            drs2 = self.interpret(ex.drs2)
            return DRS(drs1.refs + drs2.refs, drs1.conds + drs2.conds)
        assert False, f'{ex.__class__.__name__}: {ex}'

    def _make_atom(self, pred, *args):
        accum = DrtVariableExpression(Variable(pred))
        for arg in args:
            accum = DrtApplicationExpression(accum, DrtVariableExpression(Variable(arg)))
        return accum

    def _add_occur_indexing(self, base, ex):
        if self._occur_index and ex.sent_index is not None:
            if ex.discourse_id:
                base += '_%s' % ex.discourse_id
            base += '_s%s' % ex.sent_index
            base += '_w%s' % sorted(ex.word_indices)[0]
        return base