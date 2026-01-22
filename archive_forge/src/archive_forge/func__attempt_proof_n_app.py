from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_n_app(self, current, context, agenda, accessible_vars, atoms, debug):
    f, args = current.term.uncurry()
    for i, arg in enumerate(args):
        if not TableauProver.is_atom(arg):
            ctx = f
            nv = Variable('X%s' % _counter.get())
            for j, a in enumerate(args):
                ctx = ctx(VariableExpression(nv)) if i == j else ctx(a)
            if context:
                ctx = context(ctx).simplify()
            ctx = LambdaExpression(nv, -ctx)
            agenda.put(-arg, ctx)
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
    raise Exception('If this method is called, there must be a non-atomic argument')