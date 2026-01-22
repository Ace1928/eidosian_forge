from __future__ import print_function, with_statement
import logging
import os
import sys
import time
import bz2
import itertools
import numpy as np
import scipy.linalg
import gensim
class ClippedCorpus:

    def __init__(self, corpus, max_docs, max_terms):
        self.corpus = corpus
        self.max_docs, self.max_terms = (max_docs, max_terms)

    def __iter__(self):
        for doc in itertools.islice(self.corpus, self.max_docs):
            yield [(f, w) for f, w in doc if f < self.max_terms]