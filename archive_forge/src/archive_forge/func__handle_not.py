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
def _handle_not(self):
    self.assertToken(self.token(), '(')
    drs = self.process_next_expression(None)
    self.assertToken(self.token(), ')')
    return BoxerNot(drs)