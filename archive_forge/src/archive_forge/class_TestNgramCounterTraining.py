import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
class TestNgramCounterTraining:

    @classmethod
    def setup_class(self):
        self.counter = NgramCounter()
        self.case = unittest.TestCase()

    @pytest.mark.parametrize('case', ['', [], None])
    def test_empty_inputs(self, case):
        test = NgramCounter(case)
        assert 2 not in test
        assert test[1] == FreqDist()

    def test_train_on_unigrams(self):
        words = list('abcd')
        counter = NgramCounter([[(w,) for w in words]])
        assert not counter[3]
        assert not counter[2]
        self.case.assertCountEqual(words, counter[1].keys())

    def test_train_on_illegal_sentences(self):
        str_sent = ['Check', 'this', 'out', '!']
        list_sent = [['Check', 'this'], ['this', 'out'], ['out', '!']]
        with pytest.raises(TypeError):
            NgramCounter([str_sent])
        with pytest.raises(TypeError):
            NgramCounter([list_sent])

    def test_train_on_bigrams(self):
        bigram_sent = [('a', 'b'), ('c', 'd')]
        counter = NgramCounter([bigram_sent])
        assert not bool(counter[3])

    def test_train_on_mix(self):
        mixed_sent = [('a', 'b'), ('c', 'd'), ('e', 'f', 'g'), ('h',)]
        counter = NgramCounter([mixed_sent])
        unigrams = ['h']
        bigram_contexts = [('a',), ('c',)]
        trigram_contexts = [('e', 'f')]
        self.case.assertCountEqual(unigrams, counter[1].keys())
        self.case.assertCountEqual(bigram_contexts, counter[2].keys())
        self.case.assertCountEqual(trigram_contexts, counter[3].keys())