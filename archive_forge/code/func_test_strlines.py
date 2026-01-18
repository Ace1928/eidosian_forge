from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_strlines():
    q = 'this quote (") is in the middle'
    assert strlines(q, 10) == dedent('        (\n        \'this quo\'\n        \'te (") i\'\n        \'s in the\'\n        \' middle\'\n        )')
    assert q == 'this quote (") is in the middle'
    q = "this quote (') is in the middle"
    assert strlines(q, 20) == dedent('        (\n        "this quote (\') is "\n        "in the middle"\n        )')
    assert strlines('\\left') == "(\nr'\\left'\n)"
    assert strlines('\\left', short=True) == "r'\\left'"
    assert strlines('\\le"ft') == '(\nr\'\\le"ft\'\n)'
    q = 'this\nother line'
    assert strlines(q) == rawlines(q)