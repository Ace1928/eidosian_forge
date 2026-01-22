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
class BoxerRel(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, var1, var2, rel, sense):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.var1 = var1
        self.var2 = var2
        self.rel = rel
        self.sense = sense

    def _variables(self):
        return ({self.var1, self.var2}, set(), set())

    def clean(self):
        return BoxerRel(self.discourse_id, self.sent_index, self.word_indices, self.var1, self.var2, self._clean_name(self.rel), self.sense)

    def renumber_sentences(self, f):
        return BoxerRel(self.discourse_id, f(self.sent_index), self.word_indices, self.var1, self.var2, self.rel, self.sense)

    def __iter__(self):
        return iter((self.var1, self.var2, self.rel, self.sense))

    def _pred(self):
        return 'rel'