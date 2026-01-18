import unittest
def test_tuple_IReadSequence(self):
    from zope.interface.common.sequence import IReadSequence
    self._callFUT(IReadSequence, tuple, tentative=True)