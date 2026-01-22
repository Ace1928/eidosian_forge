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
def _make_atom(self, pred, *args):
    accum = DrtVariableExpression(Variable(pred))
    for arg in args:
        accum = DrtApplicationExpression(accum, DrtVariableExpression(Variable(arg)))
    return accum