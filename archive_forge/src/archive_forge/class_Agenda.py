from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
class Agenda:

    def __init__(self):
        self.sets = tuple((set() for i in range(21)))

    def clone(self):
        new_agenda = Agenda()
        set_list = [s.copy() for s in self.sets]
        new_allExs = set()
        for allEx, _ in set_list[Categories.ALL]:
            new_allEx = AllExpression(allEx.variable, allEx.term)
            try:
                new_allEx._used_vars = {used for used in allEx._used_vars}
            except AttributeError:
                new_allEx._used_vars = set()
            new_allExs.add((new_allEx, None))
        set_list[Categories.ALL] = new_allExs
        set_list[Categories.N_EQ] = {(NegatedExpression(n_eq.term), ctx) for n_eq, ctx in set_list[Categories.N_EQ]}
        new_agenda.sets = tuple(set_list)
        return new_agenda

    def __getitem__(self, index):
        return self.sets[index]

    def put(self, expression, context=None):
        if isinstance(expression, AllExpression):
            ex_to_add = AllExpression(expression.variable, expression.term)
            try:
                ex_to_add._used_vars = {used for used in expression._used_vars}
            except AttributeError:
                ex_to_add._used_vars = set()
        else:
            ex_to_add = expression
        self.sets[self._categorize_expression(ex_to_add)].add((ex_to_add, context))

    def put_all(self, expressions):
        for expression in expressions:
            self.put(expression)

    def put_atoms(self, atoms):
        for atom, neg in atoms:
            if neg:
                self[Categories.N_ATOM].add((-atom, None))
            else:
                self[Categories.ATOM].add((atom, None))

    def pop_first(self):
        """Pop the first expression that appears in the agenda"""
        for i, s in enumerate(self.sets):
            if s:
                if i in [Categories.N_EQ, Categories.ALL]:
                    for ex in s:
                        try:
                            if not ex[0]._exhausted:
                                s.remove(ex)
                                return (ex, i)
                        except AttributeError:
                            s.remove(ex)
                            return (ex, i)
                else:
                    return (s.pop(), i)
        return ((None, None), None)

    def replace_all(self, old, new):
        for s in self.sets:
            for ex, ctx in s:
                ex.replace(old.variable, new)
                if ctx is not None:
                    ctx.replace(old.variable, new)

    def mark_alls_fresh(self):
        for u, _ in self.sets[Categories.ALL]:
            u._exhausted = False

    def mark_neqs_fresh(self):
        for neq, _ in self.sets[Categories.N_EQ]:
            neq._exhausted = False

    def _categorize_expression(self, current):
        if isinstance(current, NegatedExpression):
            return self._categorize_NegatedExpression(current)
        elif isinstance(current, FunctionVariableExpression):
            return Categories.PROP
        elif TableauProver.is_atom(current):
            return Categories.ATOM
        elif isinstance(current, AllExpression):
            return Categories.ALL
        elif isinstance(current, AndExpression):
            return Categories.AND
        elif isinstance(current, OrExpression):
            return Categories.OR
        elif isinstance(current, ImpExpression):
            return Categories.IMP
        elif isinstance(current, IffExpression):
            return Categories.IFF
        elif isinstance(current, EqualityExpression):
            return Categories.EQ
        elif isinstance(current, ExistsExpression):
            return Categories.EXISTS
        elif isinstance(current, ApplicationExpression):
            return Categories.APP
        else:
            raise ProverParseError('cannot categorize %s' % current.__class__.__name__)

    def _categorize_NegatedExpression(self, current):
        negated = current.term
        if isinstance(negated, NegatedExpression):
            return Categories.D_NEG
        elif isinstance(negated, FunctionVariableExpression):
            return Categories.N_PROP
        elif TableauProver.is_atom(negated):
            return Categories.N_ATOM
        elif isinstance(negated, AllExpression):
            return Categories.N_ALL
        elif isinstance(negated, AndExpression):
            return Categories.N_AND
        elif isinstance(negated, OrExpression):
            return Categories.N_OR
        elif isinstance(negated, ImpExpression):
            return Categories.N_IMP
        elif isinstance(negated, IffExpression):
            return Categories.N_IFF
        elif isinstance(negated, EqualityExpression):
            return Categories.N_EQ
        elif isinstance(negated, ExistsExpression):
            return Categories.N_EXISTS
        elif isinstance(negated, ApplicationExpression):
            return Categories.N_APP
        else:
            raise ProverParseError('cannot categorize %s' % negated.__class__.__name__)