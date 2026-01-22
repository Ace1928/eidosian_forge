from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _output_logprob(self, state, symbol):
    """
        :return: the log probability of the symbol being observed in the given
            state
        :rtype: float
        """
    return self._outputs[state].logprob(symbol)