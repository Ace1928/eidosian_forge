import logging
import unittest
import numbers
from os import remove
import numpy as np
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
from gensim.matutils import jensen_shannon

Automated tests for the author-topic model (AuthorTopicModel class). These tests
are based on the unit tests of LDA; the classes are quite similar, and the tests
needed are thus quite similar.
