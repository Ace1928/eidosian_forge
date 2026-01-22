import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class UnificationException(Exception):

    def __init__(self, a, b):
        Exception.__init__(self, f"'{a}' cannot unify with '{b}'")