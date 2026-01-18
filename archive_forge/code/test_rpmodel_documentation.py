import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import rpmodel
from gensim import matutils
from gensim.test.utils import datapath, get_tmpfile

Automated tests for checking transformation algorithms (the models package).
