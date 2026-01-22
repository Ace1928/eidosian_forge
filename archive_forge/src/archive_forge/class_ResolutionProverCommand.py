import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class ResolutionProverCommand(BaseProverCommand):

    def __init__(self, goal=None, assumptions=None, prover=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        if prover is not None:
            assert isinstance(prover, ResolutionProver)
        else:
            prover = ResolutionProver()
        BaseProverCommand.__init__(self, prover, goal, assumptions)
        self._clauses = None

    def prove(self, verbose=False):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        if self._result is None:
            self._result, clauses = self._prover._prove(self.goal(), self.assumptions(), verbose)
            self._clauses = clauses
            self._proof = ResolutionProverCommand._decorate_clauses(clauses)
        return self._result

    def find_answers(self, verbose=False):
        self.prove(verbose)
        answers = set()
        answer_ex = VariableExpression(Variable(ResolutionProver.ANSWER_KEY))
        for clause in self._clauses:
            for term in clause:
                if isinstance(term, ApplicationExpression) and term.function == answer_ex and (not isinstance(term.argument, IndividualVariableExpression)):
                    answers.add(term.argument)
        return answers

    @staticmethod
    def _decorate_clauses(clauses):
        """
        Decorate the proof output.
        """
        out = ''
        max_clause_len = max((len(str(clause)) for clause in clauses))
        max_seq_len = len(str(len(clauses)))
        for i in range(len(clauses)):
            parents = 'A'
            taut = ''
            if clauses[i].is_tautology():
                taut = 'Tautology'
            if clauses[i]._parents:
                parents = str(clauses[i]._parents)
            parents = ' ' * (max_clause_len - len(str(clauses[i])) + 1) + parents
            seq = ' ' * (max_seq_len - len(str(i + 1))) + str(i + 1)
            out += f'[{seq}] {clauses[i]} {parents} {taut}\n'
        return out