import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_shortPythonVersion(self) -> None:
    """
        Verify if the Python version is returned correctly.
        """
    ver = shortPythonVersion().split('.')
    for i in range(3):
        self.assertEqual(int(ver[i]), sys.version_info[i])