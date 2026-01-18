from __future__ import with_statement
import logging
import unittest
from functools import partial
import numpy as np
from gensim import corpora, models, utils, matutils
from gensim.parsing.preprocessing import preprocess_documents, preprocess_string, DEFAULT_FILTERS
from gensim.test.utils import datapath
correlation with human data > 0.6
        (this is the value which was achieved in the original paper)
        