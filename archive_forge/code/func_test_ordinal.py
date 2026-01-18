from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_ordinal():
    assert ordinal(-1) == '-1st'
    assert ordinal(0) == '0th'
    assert ordinal(1) == '1st'
    assert ordinal(2) == '2nd'
    assert ordinal(3) == '3rd'
    assert all((ordinal(i).endswith('th') for i in range(4, 21)))
    assert ordinal(100) == '100th'
    assert ordinal(101) == '101st'
    assert ordinal(102) == '102nd'
    assert ordinal(103) == '103rd'
    assert ordinal(104) == '104th'
    assert ordinal(200) == '200th'
    assert all((ordinal(i) == str(i) + 'th' for i in range(-220, -203)))