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
class BoxerNot(AbstractBoxerDrs):

    def __init__(self, drs):
        AbstractBoxerDrs.__init__(self)
        self.drs = drs

    def _variables(self):
        return self.drs._variables()

    def atoms(self):
        return self.drs.atoms()

    def clean(self):
        return BoxerNot(self.drs.clean())

    def renumber_sentences(self, f):
        return BoxerNot(self.drs.renumber_sentences(f))

    def __repr__(self):
        return 'not(%s)' % self.drs

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.drs == other.drs

    def __ne__(self, other):
        return not self == other
    __hash__ = AbstractBoxerDrs.__hash__