from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize

        Return the first-order logic formula tree for this underspecified
        representation using the plugging given.
        