import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
Test lsi[corpus] transformation.