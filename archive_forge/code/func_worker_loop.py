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
def worker_loop():
    """Compute log probability for each sentence, lifting lists of sentences from the jobs queue."""
    work = np.zeros(1, dtype=REAL)
    neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
    while True:
        job = job_queue.get()
        if job is None:
            break
        ns = 0
        for sentence_id, sentence in job:
            if sentence_id >= total_sentences:
                break
            if self.sg:
                score = score_sentence_sg(self, sentence, work)
            else:
                score = score_sentence_cbow(self, sentence, work, neu1)
            sentence_scores[sentence_id] = score
            ns += 1
        progress_queue.put(ns)