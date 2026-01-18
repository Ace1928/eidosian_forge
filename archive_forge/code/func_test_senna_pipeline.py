import unittest
from os import environ, path, sep
from nltk.classify import Senna
from nltk.tag import SennaChunkTagger, SennaNERTagger, SennaTagger
def test_senna_pipeline(self):
    """Senna pipeline interface"""
    pipeline = Senna(SENNA_EXECUTABLE_PATH, ['pos', 'chk', 'ner'])
    sent = 'Dusseldorf is an international business center'.split()
    result = [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(sent)]
    expected = [('Dusseldorf', 'B-NP', 'B-LOC', 'NNP'), ('is', 'B-VP', 'O', 'VBZ'), ('an', 'B-NP', 'O', 'DT'), ('international', 'I-NP', 'O', 'JJ'), ('business', 'I-NP', 'O', 'NN'), ('center', 'I-NP', 'O', 'NN')]
    self.assertEqual(result, expected)