import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
class finite:

    def __init__(self, num_lines: int, **kwargs: typing.Any):
        self.num_lines = num_lines
        self.kwargs = kwargs

    def __call__(self) -> int:
        ap = autopage.AutoPager(pager_command=autopage.command.Less(), **self.kwargs)
        with ap as out:
            for i in range(self.num_lines):
                print(i, file=out)
        return ap.exit_code()