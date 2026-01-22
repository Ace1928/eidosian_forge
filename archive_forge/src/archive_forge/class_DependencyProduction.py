import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
class DependencyProduction(Production):
    """
    A dependency grammar production.  Each production maps a single
    head word to an unordered list of one or more modifier words.
    """

    def __str__(self):
        """
        Return a verbose string representation of the ``DependencyProduction``.

        :rtype: str
        """
        result = f"'{self._lhs}' ->"
        for elt in self._rhs:
            result += f" '{elt}'"
        return result