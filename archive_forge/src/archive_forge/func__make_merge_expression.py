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
def _make_merge_expression(self, sent_index, word_indices, drs1, drs2):
    return BoxerDrs(drs1.refs + drs2.refs, drs1.conds + drs2.conds)