import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def assertMatchBasenameAndFullpath(self, matchset):
    self.assertMatch(matchset)
    self.assertMatch(matchset, glob_prefix='./')