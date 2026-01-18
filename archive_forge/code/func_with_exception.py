import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def with_exception() -> int:

    class MyException(Exception):
        pass
    ap = autopage.AutoPager(pager_command=autopage.command.Less())
    try:
        with ap as out:
            for i in range(50):
                print(i, file=out)
            raise MyException()
    except MyException:
        pass
    return ap.exit_code()