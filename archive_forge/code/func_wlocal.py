import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus
from gensim.corpora import Dictionary
def wlocal(tf):
    assert isinstance(tf, np.ndarray)
    return iter(tf + 1)