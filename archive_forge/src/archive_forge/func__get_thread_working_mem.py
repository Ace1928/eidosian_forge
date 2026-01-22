from the disk or network on-the-fly, without loading your entire corpus into RAM.
from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty
from numpy import float32 as REAL
import numpy as np
from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab  # noqa
from smart_open.compression import get_supported_extensions
def _get_thread_working_mem(self):
    """Computes the memory used per worker thread.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        """
    work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
    neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
    return (work, neu1)