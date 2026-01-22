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
def _handle_whq(self):
    self.assertToken(self.token(), '(')
    self.assertToken(self.token(), '[')
    ans_types = []
    while self.token(0) != ']':
        cat = self.token()
        self.assertToken(self.token(), ':')
        if cat == 'des':
            ans_types.append(self.token())
        elif cat == 'num':
            ans_types.append('number')
            typ = self.token()
            if typ == 'cou':
                ans_types.append('count')
            else:
                ans_types.append(typ)
        else:
            ans_types.append(self.token())
    self.token()
    self.assertToken(self.token(), ',')
    d1 = self.process_next_expression(None)
    self.assertToken(self.token(), ',')
    ref = self.parse_variable()
    self.assertToken(self.token(), ',')
    d2 = self.process_next_expression(None)
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: BoxerWhq(self.discourse_id, sent_index, word_indices, ans_types, d1, ref, d2)