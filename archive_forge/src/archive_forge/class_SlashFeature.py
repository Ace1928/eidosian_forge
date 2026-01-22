import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class SlashFeature(Feature):

    def read_value(self, s, position, reentrances, parser):
        return parser.read_partial(s, position, reentrances)