import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class TruthValueType(BasicType):

    def __str__(self):
        return 't'

    def str(self):
        return 'BOOL'