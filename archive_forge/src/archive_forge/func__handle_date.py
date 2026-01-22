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
def _handle_date(self, arg):
    conds = []
    (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
    self.assertToken(self.token(), '(')
    pol = self.token()
    self.assertToken(self.token(), ')')
    conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_pol_{pol}', 'a', 0))
    self.assertToken(self.token(), ',')
    (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
    year = self.token()
    if year != 'XXXX':
        year = year.replace(':', '_')
        conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_year_{year}', 'a', 0))
    self.assertToken(self.token(), ',')
    (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
    month = self.token()
    if month != 'XX':
        conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_month_{month}', 'a', 0))
    self.assertToken(self.token(), ',')
    (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
    day = self.token()
    if day != 'XX':
        conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_day_{day}', 'a', 0))
    return conds