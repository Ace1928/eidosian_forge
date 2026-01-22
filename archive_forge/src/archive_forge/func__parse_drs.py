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
def _parse_drs(self, drs_string, discourse_id, use_disc_id):
    return BoxerOutputDrsParser([None, discourse_id][use_disc_id]).parse(drs_string)