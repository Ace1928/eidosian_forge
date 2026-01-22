import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
class TestBLEUWithBadSentence(unittest.TestCase):

    def test_corpus_bleu_with_bad_sentence(self):
        hyp = 'Teo S yb , oe uNb , R , T t , , t Tue Ar saln S , , 5istsi l , 5oe R ulO sae oR R'
        ref = str('Their tasks include changing a pump on the faulty stokehold .Likewise , two species that are very similar in morphology were distinguished using genetics .')
        references = [[ref.split()]]
        hypotheses = [hyp.split()]
        try:
            with self.assertWarns(UserWarning):
                self.assertAlmostEqual(corpus_bleu(references, hypotheses), 0.0, places=4)
        except AttributeError:
            self.assertAlmostEqual(corpus_bleu(references, hypotheses), 0.0, places=4)