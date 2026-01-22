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
def _ensure_shard(self, offset):
    if self.current_shard is None:
        shard_n = self.shard_by_offset(offset)
        self.load_shard(shard_n)
    elif not self.in_current(offset):
        if self.in_next(offset):
            self.load_shard(self.current_shard_n + 1)
        else:
            shard_n = self.shard_by_offset(offset)
            self.load_shard(shard_n)