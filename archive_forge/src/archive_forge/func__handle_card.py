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
def _handle_card(self):
    self.assertToken(self.token(), '(')
    variable = self.parse_variable()
    self.assertToken(self.token(), ',')
    value = self.token()
    self.assertToken(self.token(), ',')
    type = self.token()
    self.assertToken(self.token(), ')')
    return lambda sent_index, word_indices: BoxerCard(self.discourse_id, sent_index, word_indices, variable, value, type)