from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def validate_sig_len(self, new_sig):
    if self.signature_len is None:
        self.signature_len = len(new_sig)
    elif self.signature_len != len(new_sig):
        raise Exception('Signature lengths do not match')