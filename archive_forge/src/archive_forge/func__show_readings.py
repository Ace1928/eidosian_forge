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
def _show_readings(self, sentence=None):
    """
        Print out the readings for  the discourse (or a single sentence).
        """
    if sentence is not None:
        print("The sentence '%s' has these readings:" % sentence)
        for r in [str(reading) for reading in self._get_readings(sentence)]:
            print('    %s' % r)
    else:
        for sid in sorted(self._readings):
            print()
            print('%s readings:' % sid)
            print()
            for rid in sorted(self._readings[sid]):
                lf = self._readings[sid][rid]
                print(f'{rid}: {lf.normalize()}')