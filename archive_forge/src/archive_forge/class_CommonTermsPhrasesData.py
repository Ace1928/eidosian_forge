import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class CommonTermsPhrasesData:
    """This mixin permits to reuse tests with the connector_words option."""
    sentences = [['human', 'interface', 'with', 'computer'], ['survey', 'of', 'user', 'computer', 'system', 'lack', 'of', 'interest'], ['eps', 'user', 'interface', 'system'], ['system', 'and', 'human', 'system', 'eps'], ['user', 'lack', 'of', 'interest'], ['trees'], ['graph', 'of', 'trees'], ['data', 'and', 'graph', 'of', 'trees'], ['data', 'and', 'graph', 'survey'], ['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    connector_words = ['of', 'and', 'for']
    bigram1 = u'lack_of_interest'
    bigram2 = u'data_and_graph'
    bigram3 = u'human_interface'
    expression1 = u'lack of interest'
    expression2 = u'data and graph'
    expression3 = u'human interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)