import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_transform_autocall():
    for case in [AUTOCALL_QUOTE, AUTOCALL_QUOTE2, AUTOCALL_PAREN]:
        print('Testing %r' % case[0])
        check_transform(ipt2.EscapedCommand, case)