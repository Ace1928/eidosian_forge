import os
import unittest
import pytest
from monty.io import (
def test_reverse_readline_fake_big(self):
    """
        Make sure that large textfiles are read properly
        """
    with open(os.path.join(test_dir, '3000_lines.txt')) as f:
        for idx, line in enumerate(reverse_readline(f, max_mem=0)):
            assert int(line) == self.NUMLINES - idx, 'read_backwards read {} whereas it should '('have read {}').format(int(line), self.NUMLINES - idx)