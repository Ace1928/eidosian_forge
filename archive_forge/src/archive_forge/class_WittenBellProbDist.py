import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class WittenBellProbDist(ProbDistI):
    """
    The Witten-Bell estimate of a probability distribution. This distribution
    allocates uniform probability mass to as yet unseen events by using the
    number of events that have only been seen once. The probability mass
    reserved for unseen events is equal to *T / (N + T)*
    where *T* is the number of observed event types and *N* is the total
    number of observed events. This equates to the maximum likelihood estimate
    of a new type event occurring. The remaining probability mass is discounted
    such that all probability estimates sum to one, yielding:

        - *p = T / Z (N + T)*, if count = 0
        - *p = c / (N + T)*, otherwise
    """

    def __init__(self, freqdist, bins=None):
        """
        Creates a distribution of Witten-Bell probability estimates.  This
        distribution allocates uniform probability mass to as yet unseen
        events by using the number of events that have only been seen once. The
        probability mass reserved for unseen events is equal to *T / (N + T)*
        where *T* is the number of observed event types and *N* is the total
        number of observed events. This equates to the maximum likelihood
        estimate of a new type event occurring. The remaining probability mass
        is discounted such that all probability estimates sum to one,
        yielding:

            - *p = T / Z (N + T)*, if count = 0
            - *p = c / (N + T)*, otherwise

        The parameters *T* and *N* are taken from the ``freqdist`` parameter
        (the ``B()`` and ``N()`` values). The normalizing factor *Z* is
        calculated using these values along with the ``bins`` parameter.

        :param freqdist: The frequency counts upon which to base the
            estimation.
        :type freqdist: FreqDist
        :param bins: The number of possible event types. This must be at least
            as large as the number of bins in the ``freqdist``. If None, then
            it's assumed to be equal to that of the ``freqdist``
        :type bins: int
        """
        assert bins is None or bins >= freqdist.B(), 'bins parameter must not be less than %d=freqdist.B()' % freqdist.B()
        if bins is None:
            bins = freqdist.B()
        self._freqdist = freqdist
        self._T = self._freqdist.B()
        self._Z = bins - self._freqdist.B()
        self._N = self._freqdist.N()
        if self._N == 0:
            self._P0 = 1.0 / self._Z
        else:
            self._P0 = self._T / (self._Z * (self._N + self._T))

    def prob(self, sample):
        c = self._freqdist[sample]
        return c / (self._N + self._T) if c != 0 else self._P0

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def freqdist(self):
        return self._freqdist

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return '<WittenBellProbDist based on %d samples>' % self._freqdist.N()