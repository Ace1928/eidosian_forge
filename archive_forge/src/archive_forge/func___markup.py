import os
import pytest
from bs4 import (
def __markup(self, filename):
    if not filename.endswith(self.TESTCASE_SUFFIX):
        filename += self.TESTCASE_SUFFIX
    this_dir = os.path.split(__file__)[0]
    path = os.path.join(this_dir, 'fuzz', filename)
    return open(path, 'rb').read()