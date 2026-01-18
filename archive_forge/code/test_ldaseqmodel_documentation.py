import unittest
import logging
import numpy as np  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath


Tests to check DTM math functions and Topic-Word, Doc-Topic proportions.

