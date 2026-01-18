import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI

        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        