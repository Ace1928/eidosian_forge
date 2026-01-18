import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_supportsThreads(self) -> None:
    """
        L{Platform.supportsThreads} returns C{True} if threads can be created in
        this runtime, C{False} otherwise.
        """
    try:
        namedModule('threading')
    except ImportError:
        self.assertFalse(Platform().supportsThreads())
    else:
        self.assertTrue(Platform().supportsThreads())