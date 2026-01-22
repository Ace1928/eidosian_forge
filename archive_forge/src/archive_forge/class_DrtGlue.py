import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
class DrtGlue(Glue):

    def __init__(self, semtype_file=None, remove_duplicates=False, depparser=None, verbose=False):
        if not semtype_file:
            semtype_file = os.path.join('grammars', 'sample_grammars', 'drt_glue.semtype')
        Glue.__init__(self, semtype_file, remove_duplicates, depparser, verbose)

    def get_glue_dict(self):
        return DrtGlueDict(self.semtype_file)