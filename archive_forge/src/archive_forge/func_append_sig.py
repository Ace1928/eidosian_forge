from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def append_sig(self, new_sig):
    self.validate_sig_len(new_sig)
    self.signatures.append(new_sig)