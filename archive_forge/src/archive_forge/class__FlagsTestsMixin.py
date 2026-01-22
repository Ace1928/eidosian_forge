import warnings
from twisted.trial.unittest import TestCase
class _FlagsTestsMixin:
    """
    Mixin defining setup code for any tests for L{Flags} subclasses.

    @ivar FXF: A L{Flags} subclass created for each test method.
    """

    def setUp(self):
        """
        Create a fresh new L{Flags} subclass for each unit test to use.  Since
        L{Flags} is stateful, re-using the same subclass across test methods
        makes exercising all of the implementation code paths difficult.
        """

        class FXF(Flags):
            READ = FlagConstant()
            WRITE = FlagConstant()
            APPEND = FlagConstant()
            EXCLUSIVE = FlagConstant(32)
            TEXT = FlagConstant()
        self.FXF = FXF