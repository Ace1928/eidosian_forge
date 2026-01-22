import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
class DependencyGrammar:
    """
    A dependency grammar.  A DependencyGrammar consists of a set of
    productions.  Each production specifies a head/modifier relationship
    between a pair of words.
    """

    def __init__(self, productions):
        """
        Create a new dependency grammar, from the set of ``Productions``.

        :param productions: The list of productions that defines the grammar
        :type productions: list(Production)
        """
        self._productions = productions

    @classmethod
    def fromstring(cls, input):
        productions = []
        for linenum, line in enumerate(input.split('\n')):
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            try:
                productions += _read_dependency_production(line)
            except ValueError as e:
                raise ValueError(f'Unable to parse line {linenum}: {line}') from e
        if len(productions) == 0:
            raise ValueError('No productions found!')
        return cls(productions)

    def contains(self, head, mod):
        """
        :param head: A head word.
        :type head: str
        :param mod: A mod word, to test as a modifier of 'head'.
        :type mod: str

        :return: true if this ``DependencyGrammar`` contains a
            ``DependencyProduction`` mapping 'head' to 'mod'.
        :rtype: bool
        """
        for production in self._productions:
            for possibleMod in production._rhs:
                if production._lhs == head and possibleMod == mod:
                    return True
        return False

    def __contains__(self, head_mod):
        """
        Return True if this ``DependencyGrammar`` contains a
        ``DependencyProduction`` mapping 'head' to 'mod'.

        :param head_mod: A tuple of a head word and a mod word,
            to test as a modifier of 'head'.
        :type head: Tuple[str, str]
        :rtype: bool
        """
        try:
            head, mod = head_mod
        except ValueError as e:
            raise ValueError("Must use a tuple of strings, e.g. `('price', 'of') in grammar`") from e
        return self.contains(head, mod)

    def __str__(self):
        """
        Return a verbose string representation of the ``DependencyGrammar``

        :rtype: str
        """
        str = 'Dependency grammar with %d productions' % len(self._productions)
        for production in self._productions:
            str += '\n  %s' % production
        return str

    def __repr__(self):
        """
        Return a concise string representation of the ``DependencyGrammar``
        """
        return 'Dependency grammar with %d productions' % len(self._productions)