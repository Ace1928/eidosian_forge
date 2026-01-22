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

        Multiply every thread in ``discourse`` by every reading in ``readings``.

        Given discourse = [['A'], ['B']], readings = ['a', 'b', 'c'] , returns
        [['A', 'a'], ['A', 'b'], ['A', 'c'], ['B', 'a'], ['B', 'b'], ['B', 'c']]

        :param discourse: the current list of readings
        :type discourse: list of lists
        :param readings: an additional list of readings
        :type readings: list(Expression)
        :rtype: A list of lists
        