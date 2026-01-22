import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def _handle_timex(self):
    self.assertToken(self.token(), '(')
    arg = self.parse_variable()
    self.assertToken(self.token(), ',')
    new_conds = self._handle_time_expression(arg)
    self.assertToken(self.token(), ')')
    return new_conds