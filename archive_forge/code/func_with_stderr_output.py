import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def with_stderr_output() -> int:
    ap = autopage.AutoPager(pager_command=autopage.command.Less())
    with ap as out:
        for i in range(50):
            print(i, file=out)
    print('Hello world', file=sys.stderr)
    return ap.exit_code()