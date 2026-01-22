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
class Text8Corpus:

    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        """Iterate over sentences from the "text8" corpus, unzipped from https://mattmahoney.net/dc/text8.zip."""
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        sentence, rest = ([], b'')
        with utils.open(self.fname, 'rb') as fin:
            while True:
                text = rest + fin.read(8192)
                if text == rest:
                    words = utils.to_unicode(text).split()
                    sentence.extend(words)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')
                words, rest = (utils.to_unicode(text[:last_token]).split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]