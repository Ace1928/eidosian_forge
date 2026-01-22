import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
class TrigramAssocMeasures(NgramAssocMeasures):
    """
    A collection of trigram association measures. Each association measure
    is provided as a function with four arguments::

        trigram_score_fn(n_iii,
                         (n_iix, n_ixi, n_xii),
                         (n_ixx, n_xix, n_xxi),
                         n_xxx)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

    - n_iii counts ``(w1, w2, w3)``, i.e. the trigram being scored
    - n_ixx counts ``(w1, *, *)``
    - n_xxx counts ``(*, *, *)``, i.e. any trigram
    """
    _n = 3

    @staticmethod
    def _contingency(n_iii, n_iix_tuple, n_ixx_tuple, n_xxx):
        """Calculates values of a trigram contingency table (or cube) from
        marginal values.
        >>> TrigramAssocMeasures._contingency(1, (1, 1, 1), (1, 73, 1), 2000)
        (1, 0, 0, 0, 0, 72, 0, 1927)
        """
        n_iix, n_ixi, n_xii = n_iix_tuple
        n_ixx, n_xix, n_xxi = n_ixx_tuple
        n_oii = n_xii - n_iii
        n_ioi = n_ixi - n_iii
        n_iio = n_iix - n_iii
        n_ooi = n_xxi - n_iii - n_oii - n_ioi
        n_oio = n_xix - n_iii - n_oii - n_iio
        n_ioo = n_ixx - n_iii - n_ioi - n_iio
        n_ooo = n_xxx - n_iii - n_oii - n_ioi - n_iio - n_ooi - n_oio - n_ioo
        return (n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo)

    @staticmethod
    def _marginals(*contingency):
        """Calculates values of contingency table marginals from its values.
        >>> TrigramAssocMeasures._marginals(1, 0, 0, 0, 0, 72, 0, 1927)
        (1, (1, 1, 1), (1, 73, 1), 2000)
        """
        n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo = contingency
        return (n_iii, (n_iii + n_iio, n_iii + n_ioi, n_iii + n_oii), (n_iii + n_ioi + n_iio + n_ioo, n_iii + n_oii + n_iio + n_oio, n_iii + n_oii + n_ioi + n_ooi), sum(contingency))