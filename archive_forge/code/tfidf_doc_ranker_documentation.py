import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger

        Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        