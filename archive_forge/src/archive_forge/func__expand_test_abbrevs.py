from __future__ import print_function
from patsy.util import no_pickling
def _expand_test_abbrevs(short_subterms):
    subterms = []
    for subterm in short_subterms:
        factors = []
        for factor_name in subterm:
            assert factor_name[-1] in ('+', '-')
            factors.append(_ExpandedFactor(factor_name[-1] == '+', factor_name[:-1]))
        subterms.append(_Subterm(factors))
    return subterms