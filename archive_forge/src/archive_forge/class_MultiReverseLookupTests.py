import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
class MultiReverseLookupTests(ExampleTestBase, TestCase):
    """
    Test the multi_reverse_lookup.py example script.
    """
    exampleRelativePath = 'names/examples/multi_reverse_lookup.py'