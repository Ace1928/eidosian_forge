import logging
import unittest
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import hdpmodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, common_texts
import numpy as np

        Create ldamodel object, and check if the corresponding alphas are equal.
        