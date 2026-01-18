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

        Iterate through the document stream `corpus`, saving the documents
        as a ShardedCorpus to `fname`.

        Use this method instead of calling `save_corpus` directly.
        You may need to supply some kwargs that are used upon dataset creation
        (namely: `dim`, unless the dataset can infer the dimension from the
        given corpus).

        Ignore the parameters id2word, index_fname, progress_cnt, labels
        and metadata. They currently do nothing and are here only to
        provide a compatible method signature with superclass.

        