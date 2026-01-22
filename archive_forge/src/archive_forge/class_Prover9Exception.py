import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
class Prover9Exception(Exception):

    def __init__(self, returncode, message):
        msg = p9_return_codes[returncode]
        if message:
            msg += '\n%s' % message
        Exception.__init__(self, msg)