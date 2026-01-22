from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class BackwardCombinator(DirectedBinaryCombinator):
    """
    The backward equivalent of the ForwardCombinator class.
    """

    def __init__(self, combinator, predicate, suffix=''):
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        return self._combinator.can_combine(right, left) and self._predicate(left, right)

    def combine(self, left, right):
        yield from self._combinator.combine(right, left)

    def __str__(self):
        return f'<{self._combinator}{self._suffix}'