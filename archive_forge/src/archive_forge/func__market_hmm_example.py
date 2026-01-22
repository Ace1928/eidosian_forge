from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _market_hmm_example():
    """
    Return an example HMM (described at page 381, Huang et al)
    """
    states = ['bull', 'bear', 'static']
    symbols = ['up', 'down', 'unchanged']
    A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]], np.float64)
    B = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]], np.float64)
    pi = np.array([0.5, 0.2, 0.3], np.float64)
    model = _create_hmm_tagger(states, symbols, A, B, pi)
    return (model, states, symbols)