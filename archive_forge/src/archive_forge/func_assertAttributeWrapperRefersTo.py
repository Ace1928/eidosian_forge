import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def assertAttributeWrapperRefersTo(self, attributeWrapper, fqpn, obj):
    """
        Assert that a L{twisted.python.modules.PythonAttribute} refers to a
        particular Python object.
        """
    self.assertIsInstance(attributeWrapper, self.PythonAttribute)
    self.assertEqual(attributeWrapper.name, fqpn)
    self.assertIs(attributeWrapper.load(), obj)