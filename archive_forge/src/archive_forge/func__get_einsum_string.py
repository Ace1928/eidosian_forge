from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _get_einsum_string(self, subranks, contraction_indices):
    letters = self._get_letter_generator_for_einsum()
    contraction_string = ''
    counter = 0
    d = {j: min(i) for i in contraction_indices for j in i}
    indices = []
    for rank_arg in subranks:
        lindices = []
        for i in range(rank_arg):
            if counter in d:
                lindices.append(d[counter])
            else:
                lindices.append(counter)
            counter += 1
        indices.append(lindices)
    mapping = {}
    letters_free = []
    letters_dum = []
    for i in indices:
        for j in i:
            if j not in mapping:
                l = next(letters)
                mapping[j] = l
            else:
                l = mapping[j]
            contraction_string += l
            if j in d:
                if l not in letters_dum:
                    letters_dum.append(l)
            else:
                letters_free.append(l)
        contraction_string += ','
    contraction_string = contraction_string[:-1]
    return (contraction_string, letters_free, letters_dum)