from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def _shard_name(self, n):
    """Generate the name for the n-th shard."""
    return self.output_prefix + '.' + str(n)