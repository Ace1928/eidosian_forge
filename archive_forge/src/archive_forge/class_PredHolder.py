from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
class PredHolder:
    """
    This class will be used by a dictionary that will store information
    about predicates to be used by the ``ClosedWorldProver``.

    The 'signatures' property is a list of tuples defining signatures for
    which the predicate is true.  For instance, 'see(john, mary)' would be
    result in the signature '(john,mary)' for 'see'.

    The second element of the pair is a list of pairs such that the first
    element of the pair is a tuple of variables and the second element is an
    expression of those variables that makes the predicate true.  For instance,
    'all x.all y.(see(x,y) -> know(x,y))' would result in "((x,y),('see(x,y)'))"
    for 'know'.
    """

    def __init__(self):
        self.signatures = []
        self.properties = []
        self.signature_len = None

    def append_sig(self, new_sig):
        self.validate_sig_len(new_sig)
        self.signatures.append(new_sig)

    def append_prop(self, new_prop):
        self.validate_sig_len(new_prop[0])
        self.properties.append(new_prop)

    def validate_sig_len(self, new_sig):
        if self.signature_len is None:
            self.signature_len = len(new_sig)
        elif self.signature_len != len(new_sig):
            raise Exception('Signature lengths do not match')

    def __str__(self):
        return f'({self.signatures},{self.properties},{self.signature_len})'

    def __repr__(self):
        return '%s' % self