from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """

    def __init__(self, rules):
        """Compile rules into internal lookup tables"""
        if isinstance(rules, str):
            rules = rules.splitlines()
        P = Prover()
        for rule in rules:
            a, op, b = rule.split(None, 2)
            a = Logic.fromstring(a)
            b = Logic.fromstring(b)
            if op == '->':
                P.process_rule(a, b)
            elif op == '==':
                P.process_rule(a, b)
                P.process_rule(b, a)
            else:
                raise ValueError('unknown op %r' % op)
        self.beta_rules = []
        for bcond, bimpl in P.rules_beta:
            self.beta_rules.append(({_as_pair(a) for a in bcond.args}, _as_pair(bimpl)))
        impl_a = deduce_alpha_implications(P.rules_alpha)
        impl_ab = apply_beta_to_alpha_route(impl_a, P.rules_beta)
        self.defined_facts = {_base_fact(k) for k in impl_ab.keys()}
        full_implications = defaultdict(set)
        beta_triggers = defaultdict(set)
        for k, (impl, betaidxs) in impl_ab.items():
            full_implications[_as_pair(k)] = {_as_pair(i) for i in impl}
            beta_triggers[_as_pair(k)] = betaidxs
        self.full_implications = full_implications
        self.beta_triggers = beta_triggers
        prereq = defaultdict(set)
        rel_prereq = rules_2prereq(full_implications)
        for k, pitems in rel_prereq.items():
            prereq[k] |= pitems
        self.prereq = prereq

    def _to_python(self) -> str:
        """ Generate a string with plain python representation of the instance """
        return '\n'.join(self.print_rules())

    @classmethod
    def _from_python(cls, data: dict):
        """ Generate an instance from the plain python representation """
        self = cls('')
        for key in ['full_implications', 'beta_triggers', 'prereq']:
            d = defaultdict(set)
            d.update(data[key])
            setattr(self, key, d)
        self.beta_rules = data['beta_rules']
        self.defined_facts = set(data['defined_facts'])
        return self

    def _defined_facts_lines(self):
        yield 'defined_facts = ['
        for fact in sorted(self.defined_facts):
            yield f'    {fact!r},'
        yield '] # defined_facts'

    def _full_implications_lines(self):
        yield 'full_implications = dict( ['
        for fact in sorted(self.defined_facts):
            for value in (True, False):
                yield f'    # Implications of {fact} = {value}:'
                yield f'    (({fact!r}, {value!r}), set( ('
                implications = self.full_implications[fact, value]
                for implied in sorted(implications):
                    yield f'        {implied!r},'
                yield '       ) ),'
                yield '     ),'
        yield ' ] ) # full_implications'

    def _prereq_lines(self):
        yield 'prereq = {'
        yield ''
        for fact in sorted(self.prereq):
            yield f'    # facts that could determine the value of {fact}'
            yield f'    {fact!r}: {{'
            for pfact in sorted(self.prereq[fact]):
                yield f'        {pfact!r},'
            yield '    },'
            yield ''
        yield '} # prereq'

    def _beta_rules_lines(self):
        reverse_implications = defaultdict(list)
        for n, (pre, implied) in enumerate(self.beta_rules):
            reverse_implications[implied].append((pre, n))
        yield '# Note: the order of the beta rules is used in the beta_triggers'
        yield 'beta_rules = ['
        yield ''
        m = 0
        indices = {}
        for implied in sorted(reverse_implications):
            fact, value = implied
            yield f'    # Rules implying {fact} = {value}'
            for pre, n in reverse_implications[implied]:
                indices[n] = m
                m += 1
                setstr = ', '.join(map(str, sorted(pre)))
                yield f'    ({{{setstr}}},'
                yield f'        {implied!r}),'
            yield ''
        yield '] # beta_rules'
        yield 'beta_triggers = {'
        for query in sorted(self.beta_triggers):
            fact, value = query
            triggers = [indices[n] for n in self.beta_triggers[query]]
            yield f'    {query!r}: {triggers!r},'
        yield '} # beta_triggers'

    def print_rules(self) -> Iterator[str]:
        """ Returns a generator with lines to represent the facts and rules """
        yield from self._defined_facts_lines()
        yield ''
        yield ''
        yield from self._full_implications_lines()
        yield ''
        yield ''
        yield from self._prereq_lines()
        yield ''
        yield ''
        yield from self._beta_rules_lines()
        yield ''
        yield ''
        yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
        yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"