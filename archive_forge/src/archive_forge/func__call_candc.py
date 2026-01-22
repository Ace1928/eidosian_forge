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
def _call_candc(self, inputs, discourse_ids, question, verbose=False):
    """
        Call the ``candc`` binary with the given input.

        :param inputs: list of list of str Input discourses to parse
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :param filename: str A filename for the output file
        :return: stdout
        """
    args = ['--models', os.path.join(self._candc_models_path, ['boxer', 'questions'][question]), '--candc-printer', 'boxer']
    return self._call('\n'.join(sum(([f"<META>'{id}'"] + d for d, id in zip(inputs, discourse_ids)), [])), self._candc_bin, args, verbose)