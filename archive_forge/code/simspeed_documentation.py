import logging
import sys
import itertools
import os
import math
from time import time
import numpy as np
import gensim

USAGE: %(program)s CORPUS_DENSE.mm CORPUS_SPARSE.mm [NUMDOCS]
    Run speed test of similarity queries. Only use the first NUMDOCS documents of each corpus for testing (or use all if no NUMDOCS is given).
    The two sample corpora can be downloaded from http://nlp.fi.muni.cz/projekty/gensim/wikismall.tgz

Example: ./simspeed.py wikismall.dense.mm wikismall.sparse.mm 5000
