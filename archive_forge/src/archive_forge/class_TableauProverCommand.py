from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
class TableauProverCommand(BaseProverCommand):

    def __init__(self, goal=None, assumptions=None, prover=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        if prover is not None:
            assert isinstance(prover, TableauProver)
        else:
            prover = TableauProver()
        BaseProverCommand.__init__(self, prover, goal, assumptions)