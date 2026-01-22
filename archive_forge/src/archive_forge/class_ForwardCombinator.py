from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class ForwardCombinator(DirectedBinaryCombinator):
    """
    Class representing combinators where the primary functor is on the left.

    Takes an undirected combinator, and a predicate which adds constraints
    restricting the cases in which it may apply.
    """

    def __init__(self, combinator, predicate, suffix=''):
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        return self._combinator.can_combine(left, right) and self._predicate(left, right)

    def combine(self, left, right):
        yield from self._combinator.combine(left, right)

    def __str__(self):
        return f'>{self._combinator}{self._suffix}'