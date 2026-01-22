from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class UndirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Abstract class for representing a binary combinator.
    Merely defines functions for checking if the function and argument
    are able to be combined, and what the resulting category is.

    Note that as no assumptions are made as to direction, the unrestricted
    combinators can perform all backward, forward and crossed variations
    of the combinators; these restrictions must be added in the rule
    class.
    """

    @abstractmethod
    def can_combine(self, function, argument):
        pass

    @abstractmethod
    def combine(self, function, argument):
        pass