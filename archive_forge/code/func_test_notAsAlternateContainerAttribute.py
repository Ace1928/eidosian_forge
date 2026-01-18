import warnings
from twisted.trial.unittest import TestCase
def test_notAsAlternateContainerAttribute(self):
    """
        It is explicitly disallowed (via a L{ValueError}) to use a constant
        defined on a L{Names} subclass as the value of an attribute of another
        L{Names} subclass.
        """

    def defineIt():

        class AnotherNames(Names):
            something = self.METHOD.GET
    exc = self.assertRaises(ValueError, defineIt)
    self.assertEqual('Cannot use <METHOD=GET> as the value of an attribute on AnotherNames', str(exc))