import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_null_cleanup_transformer():
    manager = ipt2.TransformerManager()
    manager.cleanup_transforms.insert(0, null_cleanup_transformer)
    assert manager.transform_cell('') == ''