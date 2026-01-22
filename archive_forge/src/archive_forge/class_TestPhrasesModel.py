import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class TestPhrasesModel(PhrasesCommon, unittest.TestCase):

    def test_export_phrases(self):
        """Test Phrases bigram and trigram export phrases."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        trigram = Phrases(bigram[self.sentences], min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.export_phrases().keys())
        seen_trigrams = set(trigram.export_phrases().keys())
        assert seen_bigrams == set(['human interface', 'response time', 'graph minors', 'minors survey'])
        assert seen_trigrams == set(['human interface', 'graph minors survey'])

    def test_find_phrases(self):
        """Test Phrases bigram find phrases."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.find_phrases(self.sentences).keys())
        assert seen_bigrams == set(['response time', 'graph minors', 'human interface'])

    def test_multiple_bigrams_single_entry(self):
        """Test a single entry produces multiple bigrams."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_bigrams = set(bigram.find_phrases(test_sentences).keys())
        assert seen_bigrams == {'graph minors', 'human interface'}

    def test_scoring_default(self):
        """Test the default scoring, from the mikolov word2vec paper."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == {5.167, 3.444}

    def test__getitem__(self):
        """Test Phrases[sentences] with a single sentence."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())
        assert phrased_sentence == ['graph_minors', 'survey', 'human_interface']

    def test_scoring_npmi(self):
        """Test normalized pointwise mutual information scoring."""
        bigram = Phrases(self.sentences, min_count=1, threshold=0.5, scoring='npmi')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == {0.882, 0.714}

    def test_custom_scorer(self):
        """Test using a custom scoring function."""
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = list(bigram.find_phrases(test_sentences).values())
        assert all((score == 1 for score in seen_scores))
        assert len(seen_scores) == 3

    def test_bad_parameters(self):
        """Test the phrases module with bad parameters."""
        self.assertRaises(ValueError, Phrases, self.sentences, min_count=0)
        self.assertRaises(ValueError, Phrases, self.sentences, threshold=-1)

    def test_pruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(self.sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)