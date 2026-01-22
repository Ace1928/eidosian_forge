import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
class ProbabilisticDependencyGrammar:
    """ """

    def __init__(self, productions, events, tags):
        self._productions = productions
        self._events = events
        self._tags = tags

    def contains(self, head, mod):
        """
        Return True if this ``DependencyGrammar`` contains a
        ``DependencyProduction`` mapping 'head' to 'mod'.

        :param head: A head word.
        :type head: str
        :param mod: A mod word, to test as a modifier of 'head'.
        :type mod: str
        :rtype: bool
        """
        for production in self._productions:
            for possibleMod in production._rhs:
                if production._lhs == head and possibleMod == mod:
                    return True
        return False

    def __str__(self):
        """
        Return a verbose string representation of the ``ProbabilisticDependencyGrammar``

        :rtype: str
        """
        str = 'Statistical dependency grammar with %d productions' % len(self._productions)
        for production in self._productions:
            str += '\n  %s' % production
        str += '\nEvents:'
        for event in self._events:
            str += '\n  %d:%s' % (self._events[event], event)
        str += '\nTags:'
        for tag_word in self._tags:
            str += f'\n {tag_word}:\t({self._tags[tag_word]})'
        return str

    def __repr__(self):
        """
        Return a concise string representation of the ``ProbabilisticDependencyGrammar``
        """
        return 'Statistical Dependency grammar with %d productions' % len(self._productions)