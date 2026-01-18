import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_transform_help():
    tf = ipt2.HelpEnd((1, 0), (1, 9))
    assert tf.transform(HELP_IN_EXPR[0]) == HELP_IN_EXPR[2]
    tf = ipt2.HelpEnd((1, 0), (2, 3))
    assert tf.transform(HELP_CONTINUED_LINE[0]) == HELP_CONTINUED_LINE[2]
    tf = ipt2.HelpEnd((1, 0), (2, 8))
    assert tf.transform(HELP_MULTILINE[0]) == HELP_MULTILINE[2]
    tf = ipt2.HelpEnd((1, 0), (1, 0))
    assert tf.transform(HELP_UNICODE[0]) == HELP_UNICODE[2]