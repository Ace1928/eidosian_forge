import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def assertContainsInOrder(self, strings, target):
    """Asserts that the strings provided are found in the target in order.

    This may be useful for checking HTML output.

    Args:
      strings: A list of strings, such as [ 'fox', 'dog' ]
      target: A target string in which to look for the strings, such as
        'The quick brown fox jumped over the lazy dog'.
    """
    if not isinstance(strings, list):
        strings = [strings]
    current_index = 0
    last_string = None
    for string in strings:
        index = target.find(str(string), current_index)
        if index == -1 and current_index == 0:
            self.fail("Did not find '%s' in '%s'" % (string, target))
        elif index == -1:
            self.fail("Did not find '%s' after '%s' in '%s'" % (string, last_string, target))
        last_string = string
        current_index = index