import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
@skip_win32
def test_xmode_skip():
    """that xmode skip frames

    Not as a doctest as pytest does not run doctests.
    """
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.expect_exact('In [1]')
    block = dedent('\n    def f():\n        __tracebackhide__ = True\n        g()\n\n    def g():\n        raise ValueError\n\n    f()\n    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact('skipping')
    block = dedent('\n    def f():\n        __tracebackhide__ = True\n        g()\n\n    def g():\n        from IPython.core.debugger import set_trace\n        set_trace()\n\n    f()\n    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect('ipdb>')
    child.sendline('w')
    child.expect('hidden')
    child.expect('ipdb>')
    child.sendline('skip_hidden false')
    child.sendline('w')
    child.expect('__traceba')
    child.expect('ipdb>')
    child.close()