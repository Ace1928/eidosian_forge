import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
def test_raiseOnBadAttributeAccess(self):
    """
        Accessing bogus attributes raises C{AttributeError}, *even*
        when that attribute actually is in the sentence data.
        """
    sentence = self.sentenceClass({'BOGUS': None})
    self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')