from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def SL2_trace(self):
    """
        Returns the simplified SL(2) trace of the Word represented by this
        object. The format of the output is a Pari polynomial in the
        variables Tw where w is a word of length 3 or less

        Examples:
        >>> Word("a").SL2_trace()
        Ta
        >>> Word("A").SL2_trace()
        Ta
        >>> Word("aa").SL2_trace()
        Ta^2 - 2
        >>> Word("abAB").SL2_trace()
        Ta^2 - Tab*Tb*Ta + (Tb^2 + (Tab^2 - 2))
        >>> Word("abca").SL2_trace()
        Tabc*Ta - Tbc
        """
    if self.letters == '':
        return pari('2')
    s = cycle_sort(self.letters)
    for L in sorted(set(s)):
        i = s.find(L)
        j = s.find(L, i + 1)
        if j > i:
            w1 = s[i:j]
            w2 = s[j:] + s[:i]
            return tr(Word(w1)) * tr(Word(w2)) - tr(Word(w1).inverse() * Word(w2))
    for i in s:
        if i.isupper():
            [w1, c, w2] = s.partition(i)
            return Word(i.lower()).SL2_trace() * Word(w2 + w1).SL2_trace() - Word(w1 + i.lower() + w2).SL2_trace()
    if len(s) >= 4:
        [x, y, z, w] = [s[0], s[1], s[2], s[3:]]
        return pari('1/2') * (tr(Word(x)) * tr(Word(y)) * tr(Word(z)) * tr(Word(w)) + tr(Word(x)) * tr(Word(y + z + w)) + tr(Word(y)) * tr(Word(x + z + w)) + tr(Word(z)) * tr(Word(x + y + w)) + tr(Word(w)) * tr(Word(x + y + z)) - tr(Word(x + z)) * tr(Word(y + w)) + tr(Word(x + w)) * tr(Word(y + z)) + tr(Word(x + y)) * tr(Word(z + w)) - tr(Word(x)) * tr(Word(y)) * tr(Word(z + w)) - tr(Word(x)) * tr(Word(w)) * tr(Word(y + z)) - tr(Word(y)) * tr(Word(z)) * tr(Word(x + w)) - tr(Word(z)) * tr(Word(w)) * tr(Word(x + y)))
    if len(s) == 3 and s != ''.join(sorted(s)):
        [x, y, z] = s
        return -Word(x + z + y).SL2_trace() + Word(x).SL2_trace() * Word(y + z).SL2_trace() + Word(y).SL2_trace() * Word(x + z).SL2_trace() + Word(z).SL2_trace() * Word(x + y).SL2_trace() - Word(x).SL2_trace() * Word(y).SL2_trace() * Word(z).SL2_trace()
    if len(s) <= 3 and s.islower():
        return pari('T' + s)