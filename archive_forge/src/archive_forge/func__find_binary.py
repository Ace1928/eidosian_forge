import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def _find_binary(self, name, verbose=False):
    binary_locations = self.binary_locations()
    if self._binary_location is not None:
        binary_locations += [self._binary_location]
    return nltk.internals.find_binary(name, searchpath=binary_locations, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)