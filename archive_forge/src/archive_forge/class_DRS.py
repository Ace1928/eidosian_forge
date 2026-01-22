import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DRS(DrtExpression, Expression):
    """A Discourse Representation Structure."""

    def __init__(self, refs, conds, consequent=None):
        """
        :param refs: list of ``DrtIndividualVariableExpression`` for the
            discourse referents
        :param conds: list of ``Expression`` for the conditions
        """
        self.refs = refs
        self.conds = conds
        self.consequent = consequent

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        if variable in self.refs:
            if not replace_bound:
                return self
            else:
                i = self.refs.index(variable)
                if self.consequent:
                    consequent = self.consequent.replace(variable, expression, True, alpha_convert)
                else:
                    consequent = None
                return DRS(self.refs[:i] + [expression.variable] + self.refs[i + 1:], [cond.replace(variable, expression, True, alpha_convert) for cond in self.conds], consequent)
        else:
            if alpha_convert:
                for ref in set(self.refs) & expression.free():
                    newvar = unique_variable(ref)
                    newvarex = DrtVariableExpression(newvar)
                    i = self.refs.index(ref)
                    if self.consequent:
                        consequent = self.consequent.replace(ref, newvarex, True, alpha_convert)
                    else:
                        consequent = None
                    self = DRS(self.refs[:i] + [newvar] + self.refs[i + 1:], [cond.replace(ref, newvarex, True, alpha_convert) for cond in self.conds], consequent)
            if self.consequent:
                consequent = self.consequent.replace(variable, expression, replace_bound, alpha_convert)
            else:
                consequent = None
            return DRS(self.refs, [cond.replace(variable, expression, replace_bound, alpha_convert) for cond in self.conds], consequent)

    def free(self):
        """:see: Expression.free()"""
        conds_free = reduce(operator.or_, [c.free() for c in self.conds], set())
        if self.consequent:
            conds_free.update(self.consequent.free())
        return conds_free - set(self.refs)

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        if recursive:
            conds_refs = self.refs + list(chain.from_iterable((c.get_refs(True) for c in self.conds)))
            if self.consequent:
                conds_refs.extend(self.consequent.get_refs(True))
            return conds_refs
        else:
            return self.refs

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        parts = list(map(function, self.conds))
        if self.consequent:
            parts.append(function(self.consequent))
        return combinator(parts)

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        consequent = function(self.consequent) if self.consequent else None
        return combinator(self.refs, list(map(function, self.conds)), consequent)

    def eliminate_equality(self):
        drs = self
        i = 0
        while i < len(drs.conds):
            cond = drs.conds[i]
            if isinstance(cond, EqualityExpression) and isinstance(cond.first, AbstractVariableExpression) and isinstance(cond.second, AbstractVariableExpression):
                drs = DRS(list(set(drs.refs) - {cond.second.variable}), drs.conds[:i] + drs.conds[i + 1:], drs.consequent)
                if cond.second.variable != cond.first.variable:
                    drs = drs.replace(cond.second.variable, cond.first, False, False)
                    i = 0
                i -= 1
            i += 1
        conds = []
        for cond in drs.conds:
            new_cond = cond.eliminate_equality()
            new_cond_simp = new_cond.simplify()
            if not isinstance(new_cond_simp, DRS) or new_cond_simp.refs or new_cond_simp.conds or new_cond_simp.consequent:
                conds.append(new_cond)
        consequent = drs.consequent.eliminate_equality() if drs.consequent else None
        return DRS(drs.refs, conds, consequent)

    def fol(self):
        if self.consequent:
            accum = None
            if self.conds:
                accum = reduce(AndExpression, [c.fol() for c in self.conds])
            if accum:
                accum = ImpExpression(accum, self.consequent.fol())
            else:
                accum = self.consequent.fol()
            for ref in self.refs[::-1]:
                accum = AllExpression(ref, accum)
            return accum
        else:
            if not self.conds:
                raise Exception('Cannot convert DRS with no conditions to FOL.')
            accum = reduce(AndExpression, [c.fol() for c in self.conds])
            for ref in map(Variable, self._order_ref_strings(self.refs)[::-1]):
                accum = ExistsExpression(ref, accum)
            return accum

    def _pretty(self):
        refs_line = ' '.join(self._order_ref_strings(self.refs))
        cond_lines = [cond for cond_line in [filter(lambda s: s.strip(), cond._pretty()) for cond in self.conds] for cond in cond_line]
        length = max([len(refs_line)] + list(map(len, cond_lines)))
        drs = [' _' + '_' * length + '_ ', '| ' + refs_line.ljust(length) + ' |', '|-' + '-' * length + '-|'] + ['| ' + line.ljust(length) + ' |' for line in cond_lines] + ['|_' + '_' * length + '_|']
        if self.consequent:
            return DrtBinaryExpression._assemble_pretty(drs, DrtTokens.IMP, self.consequent._pretty())
        return drs

    def _order_ref_strings(self, refs):
        strings = ['%s' % ref for ref in refs]
        ind_vars = []
        func_vars = []
        event_vars = []
        other_vars = []
        for s in strings:
            if is_indvar(s):
                ind_vars.append(s)
            elif is_funcvar(s):
                func_vars.append(s)
            elif is_eventvar(s):
                event_vars.append(s)
            else:
                other_vars.append(s)
        return sorted(other_vars) + sorted(event_vars, key=lambda v: int([v[2:], -1][len(v[2:]) == 0])) + sorted(func_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0]))) + sorted(ind_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0])))

    def __eq__(self, other):
        """Defines equality modulo alphabetic variance.
        If we are comparing \\x.M  and \\y.N, then check equality of M and N[x/y]."""
        if isinstance(other, DRS):
            if len(self.refs) == len(other.refs):
                converted_other = other
                for r1, r2 in zip(self.refs, converted_other.refs):
                    varex = self.make_VariableExpression(r1)
                    converted_other = converted_other.replace(r2, varex, True)
                if self.consequent == converted_other.consequent and len(self.conds) == len(converted_other.conds):
                    for c1, c2 in zip(self.conds, converted_other.conds):
                        if not c1 == c2:
                            return False
                    return True
        return False

    def __ne__(self, other):
        return not self == other
    __hash__ = Expression.__hash__

    def __str__(self):
        drs = '([{}],[{}])'.format(','.join(self._order_ref_strings(self.refs)), ', '.join(('%s' % cond for cond in self.conds)))
        if self.consequent:
            return DrtTokens.OPEN + drs + ' ' + DrtTokens.IMP + ' ' + '%s' % self.consequent + DrtTokens.CLOSE
        return drs