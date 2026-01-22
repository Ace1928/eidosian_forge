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
def _check_training_sanity(self, epochs=0, total_examples=None, total_words=None, **kwargs):
    """Checks whether the training parameters make sense.

        Parameters
        ----------
        epochs : int
            Number of training epochs. A positive integer.
        total_examples : int, optional
            Number of documents in the corpus. Either `total_examples` or `total_words` **must** be supplied.
        total_words : int, optional
            Number of words in the corpus. Either `total_examples` or `total_words` **must** be supplied.
        **kwargs : object
            Unused. Present to preserve signature among base and inherited implementations.

        Raises
        ------
        RuntimeError
            If one of the required training pre/post processing steps have not been performed.
        ValueError
            If the combination of input parameters is inconsistent.

        """
    if not self.hs and (not self.negative):
        raise ValueError("You must set either 'hs' or 'negative' to be positive for proper training. When both 'hs=0' and 'negative=0', there will be no training.")
    if self.hs and self.negative:
        logger.warning("Both hierarchical softmax and negative sampling are activated. This is probably a mistake. You should set either 'hs=0' or 'negative=0' to disable one of them. ")
    if self.alpha > self.min_alpha_yet_reached:
        logger.warning("Effective 'alpha' higher than previous training cycles")
    if not self.wv.key_to_index:
        raise RuntimeError('you must first build vocabulary before training the model')
    if not len(self.wv.vectors):
        raise RuntimeError('you must initialize vectors before training the model')
    if total_words is None and total_examples is None:
        raise ValueError("You must specify either total_examples or total_words, for proper learning-rate and progress calculations. If you've just built the vocabulary using the same corpus, using the count cached in the model is sufficient: total_examples=model.corpus_count.")
    if epochs is None or epochs <= 0:
        raise ValueError('You must specify an explicit epochs count. The usual value is epochs=model.epochs.')