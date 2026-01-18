import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_singleModuleObject(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonAttribute}
        referring to the deepest object an FQPN names, traversing one module.
        """
    import os
    self.assertAttributeWrapperRefersTo(self.wrapFQPN('os.path'), 'os.path', os.path)