import warnings
from twisted.trial.unittest import TestCase
def test_notInstantiable(self):
    """
        A subclass of L{Flags} raises L{TypeError} if an attempt is made to
        instantiate it.
        """
    self._notInstantiableTest('FXF', self.FXF)