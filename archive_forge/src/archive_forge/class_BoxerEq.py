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
class BoxerEq(BoxerIndexed):

    def __init__(self, discourse_id, sent_index, word_indices, var1, var2):
        BoxerIndexed.__init__(self, discourse_id, sent_index, word_indices)
        self.var1 = var1
        self.var2 = var2

    def _variables(self):
        return ({self.var1, self.var2}, set(), set())

    def atoms(self):
        return set()

    def renumber_sentences(self, f):
        return BoxerEq(self.discourse_id, f(self.sent_index), self.word_indices, self.var1, self.var2)

    def __iter__(self):
        return iter((self.var1, self.var2))

    def _pred(self):
        return 'eq'