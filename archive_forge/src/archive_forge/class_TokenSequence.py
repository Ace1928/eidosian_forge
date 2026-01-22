import math
import re
from nltk.tokenize.api import TokenizerI
class TokenSequence:
    """A token list with its original length and its index"""

    def __init__(self, index, wrdindex_list, original_length=None):
        original_length = original_length or len(wrdindex_list)
        self.__dict__.update(locals())
        del self.__dict__['self']