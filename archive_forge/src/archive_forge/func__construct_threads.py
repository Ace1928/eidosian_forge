import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_
from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger
def _construct_threads(self):
    """
        Use ``self._readings`` to construct a value for ``self._threads``
        and use the model builder to construct a value for ``self._filtered_threads``
        """
    thread_list = [[]]
    for sid in sorted(self._readings):
        thread_list = self.multiply(thread_list, sorted(self._readings[sid]))
    self._threads = {'d%s' % tid: thread for tid, thread in enumerate(thread_list)}
    self._filtered_threads = {}
    consistency_checked = self._check_consistency(self._threads)
    for tid, thread in self._threads.items():
        if (tid, True) in consistency_checked:
            self._filtered_threads[tid] = thread