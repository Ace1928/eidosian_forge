import os
import unittest
import pytest
from monty.io import (
def test_reverse_readline(self):
    """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
    with open(os.path.join(test_dir, '3000_lines.txt')) as f:
        for idx, line in enumerate(reverse_readline(f)):
            assert int(line) == self.NUMLINES - idx, 'read_backwards read {} whereas it should '('have read {}').format(int(line), self.NUMLINES - idx)