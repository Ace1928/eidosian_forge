import pytest
from nltk.tag import hmm
def _wikipedia_example_hmm():
    states = ['rain', 'no rain']
    symbols = ['umbrella', 'no umbrella']
    A = [[0.7, 0.3], [0.3, 0.7]]
    B = [[0.9, 0.1], [0.2, 0.8]]
    pi = [0.5, 0.5]
    seq = ['umbrella', 'umbrella', 'no umbrella', 'umbrella', 'umbrella']
    seq = list(zip(seq, [None] * len(seq)))
    model = hmm._create_hmm_tagger(states, symbols, A, B, pi)
    return (model, states, symbols, seq)