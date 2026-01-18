import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
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