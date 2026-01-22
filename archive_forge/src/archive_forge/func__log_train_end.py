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
def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed, job_tally):
    """Callback to log the end of training.

        Parameters
        ----------
        raw_word_count : int
            Number of words used in the whole training.
        trained_word_count : int
            Number of effective words used in training (after ignoring unknown words and trimming the sentence length).
        total_elapsed : int
            Total time spent during training in seconds.
        job_tally : int
            Total number of jobs processed during training.

        """
    self.add_lifecycle_event('train', msg=f'training on {raw_word_count} raw words ({trained_word_count} effective words) took {total_elapsed:.1f}s, {trained_word_count / total_elapsed:.0f} effective words/s')