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
class BoxerDrsParser(DrtParser):
    """
    Reparse the str form of subclasses of ``AbstractBoxerDrs``
    """

    def __init__(self, discourse_id=None):
        DrtParser.__init__(self)
        self.discourse_id = discourse_id

    def get_all_symbols(self):
        return [DrtTokens.OPEN, DrtTokens.CLOSE, DrtTokens.COMMA, DrtTokens.OPEN_BRACKET, DrtTokens.CLOSE_BRACKET]

    def attempt_adjuncts(self, expression, context):
        return expression

    def handle(self, tok, context):
        try:
            if tok == 'pred':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = list(map(int, self.handle_refs()))
                self.assertNextToken(DrtTokens.COMMA)
                variable = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                name = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                pos = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sense = int(self.token())
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerPred(disc_id, sent_id, word_ids, variable, name, pos, sense)
            elif tok == 'named':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = map(int, self.handle_refs())
                self.assertNextToken(DrtTokens.COMMA)
                variable = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                name = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                type = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sense = int(self.token())
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerNamed(disc_id, sent_id, word_ids, variable, name, type, sense)
            elif tok == 'rel':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = list(map(int, self.handle_refs()))
                self.assertNextToken(DrtTokens.COMMA)
                var1 = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                var2 = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                rel = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sense = int(self.token())
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerRel(disc_id, sent_id, word_ids, var1, var2, rel, sense)
            elif tok == 'prop':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = list(map(int, self.handle_refs()))
                self.assertNextToken(DrtTokens.COMMA)
                variable = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                drs = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerProp(disc_id, sent_id, word_ids, variable, drs)
            elif tok == 'not':
                self.assertNextToken(DrtTokens.OPEN)
                drs = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerNot(drs)
            elif tok == 'imp':
                self.assertNextToken(DrtTokens.OPEN)
                drs1 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.COMMA)
                drs2 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerDrs(drs1.refs, drs1.conds, drs2)
            elif tok == 'or':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = map(int, self.handle_refs())
                self.assertNextToken(DrtTokens.COMMA)
                drs1 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.COMMA)
                drs2 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerOr(disc_id, sent_id, word_ids, drs1, drs2)
            elif tok == 'eq':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = list(map(int, self.handle_refs()))
                self.assertNextToken(DrtTokens.COMMA)
                var1 = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                var2 = int(self.token())
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerEq(disc_id, sent_id, word_ids, var1, var2)
            elif tok == 'card':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = map(int, self.handle_refs())
                self.assertNextToken(DrtTokens.COMMA)
                var = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                value = self.token()
                self.assertNextToken(DrtTokens.COMMA)
                type = self.token()
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerCard(disc_id, sent_id, word_ids, var, value, type)
            elif tok == 'whq':
                self.assertNextToken(DrtTokens.OPEN)
                disc_id = self.discourse_id if self.discourse_id is not None else self.token()
                self.assertNextToken(DrtTokens.COMMA)
                sent_id = self.nullableIntToken()
                self.assertNextToken(DrtTokens.COMMA)
                word_ids = list(map(int, self.handle_refs()))
                self.assertNextToken(DrtTokens.COMMA)
                ans_types = self.handle_refs()
                self.assertNextToken(DrtTokens.COMMA)
                drs1 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.COMMA)
                var = int(self.token())
                self.assertNextToken(DrtTokens.COMMA)
                drs2 = self.process_next_expression(None)
                self.assertNextToken(DrtTokens.CLOSE)
                return BoxerWhq(disc_id, sent_id, word_ids, ans_types, drs1, var, drs2)
        except Exception as e:
            raise LogicalExpressionException(self._currentIndex, str(e)) from e
        assert False, repr(tok)

    def nullableIntToken(self):
        t = self.token()
        return int(t) if t != 'None' else None

    def get_next_token_variable(self, description):
        try:
            return self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, 'Variable expected.') from e