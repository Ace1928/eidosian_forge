import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class ProbabilisticMixIn:
    """
    A mix-in class to associate probabilities with other classes
    (trees, rules, etc.).  To use the ``ProbabilisticMixIn`` class,
    define a new class that derives from an existing class and from
    ProbabilisticMixIn.  You will need to define a new constructor for
    the new class, which explicitly calls the constructors of both its
    parent classes.  For example:

        >>> from nltk.probability import ProbabilisticMixIn
        >>> class A:
        ...     def __init__(self, x, y): self.data = (x,y)
        ...
        >>> class ProbabilisticA(A, ProbabilisticMixIn):
        ...     def __init__(self, x, y, **prob_kwarg):
        ...         A.__init__(self, x, y)
        ...         ProbabilisticMixIn.__init__(self, **prob_kwarg)

    See the documentation for the ProbabilisticMixIn
    ``constructor<__init__>`` for information about the arguments it
    expects.

    You should generally also redefine the string representation
    methods, the comparison methods, and the hashing method.
    """

    def __init__(self, **kwargs):
        """
        Initialize this object's probability.  This initializer should
        be called by subclass constructors.  ``prob`` should generally be
        the first argument for those constructors.

        :param prob: The probability associated with the object.
        :type prob: float
        :param logprob: The log of the probability associated with
            the object.
        :type logprob: float
        """
        if 'prob' in kwargs:
            if 'logprob' in kwargs:
                raise TypeError('Must specify either prob or logprob (not both)')
            else:
                ProbabilisticMixIn.set_prob(self, kwargs['prob'])
        elif 'logprob' in kwargs:
            ProbabilisticMixIn.set_logprob(self, kwargs['logprob'])
        else:
            self.__prob = self.__logprob = None

    def set_prob(self, prob):
        """
        Set the probability associated with this object to ``prob``.

        :param prob: The new probability
        :type prob: float
        """
        self.__prob = prob
        self.__logprob = None

    def set_logprob(self, logprob):
        """
        Set the log probability associated with this object to
        ``logprob``.  I.e., set the probability associated with this
        object to ``2**(logprob)``.

        :param logprob: The new log probability
        :type logprob: float
        """
        self.__logprob = logprob
        self.__prob = None

    def prob(self):
        """
        Return the probability associated with this object.

        :rtype: float
        """
        if self.__prob is None:
            if self.__logprob is None:
                return None
            self.__prob = 2 ** self.__logprob
        return self.__prob

    def logprob(self):
        """
        Return ``log(p)``, where ``p`` is the probability associated
        with this object.

        :rtype: float
        """
        if self.__logprob is None:
            if self.__prob is None:
                return None
            self.__logprob = math.log(self.__prob, 2)
        return self.__logprob