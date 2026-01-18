import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_side_effects_I():
    count = 0

    def counter(lines):
        nonlocal count
        count += 1
        return lines
    counter.has_side_effects = True
    manager = ipt2.TransformerManager()
    manager.cleanup_transforms.insert(0, counter)
    assert manager.check_complete('a=1\n') == ('complete', None)
    assert count == 0