import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class TestPhrasesModelCommonTerms(CommonTermsPhrasesData, TestPhrasesModel):
    """Test Phrases models with connector words."""

    def test_multiple_bigrams_single_entry(self):
        """Test a single entry produces multiple bigrams."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_bigrams = set(bigram.find_phrases(test_sentences).keys())
        assert seen_bigrams == set(['data and graph', 'human interface'])

    def test_find_phrases(self):
        """Test Phrases bigram export phrases."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
        seen_bigrams = set(bigram.find_phrases(self.sentences).keys())
        assert seen_bigrams == set(['human interface', 'graph of trees', 'data and graph', 'lack of interest'])

    def test_export_phrases(self):
        """Test Phrases bigram export phrases."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.export_phrases().keys())
        assert seen_bigrams == set(['and graph', 'data and', 'graph of', 'graph survey', 'human interface', 'lack of', 'of interest', 'of trees'])

    def test_scoring_default(self):
        """ test the default scoring, from the mikolov word2vec paper """
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        min_count = float(bigram.min_count)
        len_vocab = float(len(bigram.vocab))
        graph = float(bigram.vocab['graph'])
        data = float(bigram.vocab['data'])
        data_and_graph = float(bigram.vocab['data_and_graph'])
        human = float(bigram.vocab['human'])
        interface = float(bigram.vocab['interface'])
        human_interface = float(bigram.vocab['human_interface'])
        assert seen_scores == set([round((data_and_graph - min_count) / data / graph * len_vocab, 3), round((human_interface - min_count) / human / interface * len_vocab, 3)])

    def test_scoring_npmi(self):
        """Test normalized pointwise mutual information scoring."""
        bigram = Phrases(self.sentences, min_count=1, threshold=0.5, scoring='npmi', connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == set([0.74, 0.894])

    def test_custom_scorer(self):
        """Test using a custom scoring function."""
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = list(bigram.find_phrases(test_sentences).values())
        assert all(seen_scores)
        assert len(seen_scores) == 2

    def test__getitem__(self):
        """Test Phrases[sentences] with a single sentence."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())
        assert phrased_sentence == ['data_and_graph', 'survey', 'for', 'human_interface']