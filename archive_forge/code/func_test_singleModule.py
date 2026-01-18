import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_singleModule(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the single module a dotless FQPN describes.
        """
    import os
    moduleWrapper = self.wrapFQPN('os')
    self.assertIsInstance(moduleWrapper, self.PythonModule)
    self.assertIs(moduleWrapper.load(), os)