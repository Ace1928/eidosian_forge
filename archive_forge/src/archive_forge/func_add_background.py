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
def add_background(self, background, verbose=False):
    """
        Add a list of background assumptions for reasoning about the discourse.

        When called,  this method also updates the discourse model's set of readings and threads.
        :param background: Formulas which contain background information
        :type background: list(Expression)
        """
    from nltk.sem.logic import Expression
    for count, e in enumerate(background):
        assert isinstance(e, Expression)
        if verbose:
            print('Adding assumption %s to background' % count)
        self._background.append(e)
    self._construct_readings()
    self._construct_threads()