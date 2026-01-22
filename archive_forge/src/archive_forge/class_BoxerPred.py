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
class BoxerPred(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, var, name, pos, sense):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.var = var
        self.name = name
        self.pos = pos
        self.sense = sense

    def _variables(self):
        return ({self.var}, set(), set())

    def change_var(self, var):
        return BoxerPred(self.discourse_id, self.sent_index, self.word_indices, var, self.name, self.pos, self.sense)

    def clean(self):
        return BoxerPred(self.discourse_id, self.sent_index, self.word_indices, self.var, self._clean_name(self.name), self.pos, self.sense)

    def renumber_sentences(self, f):
        new_sent_index = f(self.sent_index)
        return BoxerPred(self.discourse_id, new_sent_index, self.word_indices, self.var, self.name, self.pos, self.sense)

    def __iter__(self):
        return iter((self.var, self.name, self.pos, self.sense))

    def _pred(self):
        return 'pred'