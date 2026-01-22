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
class ReadingCommand(metaclass=ABCMeta):

    @abstractmethod
    def parse_to_readings(self, sentence):
        """
        :param sentence: the sentence to read
        :type sentence: str
        """

    def process_thread(self, sentence_readings):
        """
        This method should be used to handle dependencies between readings such
        as resolving anaphora.

        :param sentence_readings: readings to process
        :type sentence_readings: list(Expression)
        :return: the list of readings after processing
        :rtype: list(Expression)
        """
        return sentence_readings

    @abstractmethod
    def combine_readings(self, readings):
        """
        :param readings: readings to combine
        :type readings: list(Expression)
        :return: one combined reading
        :rtype: Expression
        """

    @abstractmethod
    def to_fol(self, expression):
        """
        Convert this expression into a First-Order Logic expression.

        :param expression: an expression
        :type expression: Expression
        :return: a FOL version of the input expression
        :rtype: Expression
        """