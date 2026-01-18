import os
import unittest
import pytest
from monty.io import (
def test_reverse_readfile(self):
    """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
    fname = os.path.join(test_dir, '3000_lines.txt')
    for idx, line in enumerate(reverse_readfile(fname)):
        assert int(line) == self.NUMLINES - idx