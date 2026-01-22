import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _calculate_indexes(self):
    self._lhs_index = {}
    self._rhs_index = {}
    self._empty_index = {}
    self._empty_productions = []
    self._lexical_index = {}
    for prod in self._productions:
        lhs = self._get_type_if_possible(prod._lhs)
        if lhs not in self._lhs_index:
            self._lhs_index[lhs] = []
        self._lhs_index[lhs].append(prod)
        if prod._rhs:
            rhs0 = self._get_type_if_possible(prod._rhs[0])
            if rhs0 not in self._rhs_index:
                self._rhs_index[rhs0] = []
            self._rhs_index[rhs0].append(prod)
        else:
            if lhs not in self._empty_index:
                self._empty_index[lhs] = []
            self._empty_index[lhs].append(prod)
            self._empty_productions.append(prod)
        for token in prod._rhs:
            if is_terminal(token):
                self._lexical_index.setdefault(token, set()).add(prod)