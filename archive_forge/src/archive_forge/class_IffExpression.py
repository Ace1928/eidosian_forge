import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class IffExpression(BooleanExpression):
    """This class represents biconditionals"""

    def getOp(self):
        return Tokens.IFF