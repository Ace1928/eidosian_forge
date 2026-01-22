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
def _handle_eq(self):
    self.assertToken(self.token(), '(')
    var1 = self.parse_variable()
    self.assertToken(self.token(), ',')
    var2 = self.parse_variable()
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: BoxerEq(self.discourse_id, sent_index, word_indices, var1, var2)