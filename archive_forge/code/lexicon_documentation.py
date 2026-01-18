import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression

        String representation of the lexicon. Used for debugging.
        