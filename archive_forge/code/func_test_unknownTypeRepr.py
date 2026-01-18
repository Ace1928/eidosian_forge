import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
def test_unknownTypeRepr(self):
    """
        Test the repr of an empty sentence of unknown type.
        """
    sentence = self.sentenceClass({})
    expectedRepr = self._expectedRepr()
    self.assertEqual(repr(sentence), expectedRepr)