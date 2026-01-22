import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class SimpleGoodTuringProbDist(ProbDistI):
    """
    SimpleGoodTuring ProbDist approximates from frequency to frequency of
    frequency into a linear line under log space by linear regression.
    Details of Simple Good-Turing algorithm can be found in:

    - Good Turing smoothing without tears" (Gale & Sampson 1995),
      Journal of Quantitative Linguistics, vol. 2 pp. 217-237.
    - "Speech and Language Processing (Jurafsky & Martin),
      2nd Edition, Chapter 4.5 p103 (log(Nc) =  a + b*log(c))
    - https://www.grsampson.net/RGoodTur.html

    Given a set of pair (xi, yi),  where the xi denotes the frequency and
    yi denotes the frequency of frequency, we want to minimize their
    square variation. E(x) and E(y) represent the mean of xi and yi.

    - slope: b = sigma ((xi-E(x)(yi-E(y))) / sigma ((xi-E(x))(xi-E(x)))
    - intercept: a = E(y) - b.E(x)
    """
    SUM_TO_ONE = False

    def __init__(self, freqdist, bins=None):
        """
        :param freqdist: The frequency counts upon which to base the
            estimation.
        :type freqdist: FreqDist
        :param bins: The number of possible event types. This must be
            larger than the number of bins in the ``freqdist``. If None,
            then it's assumed to be equal to ``freqdist``.B() + 1
        :type bins: int
        """
        assert bins is None or bins > freqdist.B(), 'bins parameter must not be less than %d=freqdist.B()+1' % (freqdist.B() + 1)
        if bins is None:
            bins = freqdist.B() + 1
        self._freqdist = freqdist
        self._bins = bins
        r, nr = self._r_Nr()
        self.find_best_fit(r, nr)
        self._switch(r, nr)
        self._renormalize(r, nr)

    def _r_Nr_non_zero(self):
        r_Nr = self._freqdist.r_Nr()
        del r_Nr[0]
        return r_Nr

    def _r_Nr(self):
        """
        Split the frequency distribution in two list (r, Nr), where Nr(r) > 0
        """
        nonzero = self._r_Nr_non_zero()
        if not nonzero:
            return ([], [])
        return zip(*sorted(nonzero.items()))

    def find_best_fit(self, r, nr):
        """
        Use simple linear regression to tune parameters self._slope and
        self._intercept in the log-log space based on count and Nr(count)
        (Work in log space to avoid floating point underflow.)
        """
        if not r or not nr:
            return
        zr = []
        for j in range(len(r)):
            i = r[j - 1] if j > 0 else 0
            k = 2 * r[j] - i if j == len(r) - 1 else r[j + 1]
            zr_ = 2.0 * nr[j] / (k - i)
            zr.append(zr_)
        log_r = [math.log(i) for i in r]
        log_zr = [math.log(i) for i in zr]
        xy_cov = x_var = 0.0
        x_mean = sum(log_r) / len(log_r)
        y_mean = sum(log_zr) / len(log_zr)
        for x, y in zip(log_r, log_zr):
            xy_cov += (x - x_mean) * (y - y_mean)
            x_var += (x - x_mean) ** 2
        self._slope = xy_cov / x_var if x_var != 0 else 0.0
        if self._slope >= -1:
            warnings.warn('SimpleGoodTuring did not find a proper best fit line for smoothing probabilities of occurrences. The probability estimates are likely to be unreliable.')
        self._intercept = y_mean - self._slope * x_mean

    def _switch(self, r, nr):
        """
        Calculate the r frontier where we must switch from Nr to Sr
        when estimating E[Nr].
        """
        for i, r_ in enumerate(r):
            if len(r) == i + 1 or r[i + 1] != r_ + 1:
                self._switch_at = r_
                break
            Sr = self.smoothedNr
            smooth_r_star = (r_ + 1) * Sr(r_ + 1) / Sr(r_)
            unsmooth_r_star = (r_ + 1) * nr[i + 1] / nr[i]
            std = math.sqrt(self._variance(r_, nr[i], nr[i + 1]))
            if abs(unsmooth_r_star - smooth_r_star) <= 1.96 * std:
                self._switch_at = r_
                break

    def _variance(self, r, nr, nr_1):
        r = float(r)
        nr = float(nr)
        nr_1 = float(nr_1)
        return (r + 1.0) ** 2 * (nr_1 / nr ** 2) * (1.0 + nr_1 / nr)

    def _renormalize(self, r, nr):
        """
        It is necessary to renormalize all the probability estimates to
        ensure a proper probability distribution results. This can be done
        by keeping the estimate of the probability mass for unseen items as
        N(1)/N and renormalizing all the estimates for previously seen items
        (as Gale and Sampson (1995) propose). (See M&S P.213, 1999)
        """
        prob_cov = 0.0
        for r_, nr_ in zip(r, nr):
            prob_cov += nr_ * self._prob_measure(r_)
        if prob_cov:
            self._renormal = (1 - self._prob_measure(0)) / prob_cov

    def smoothedNr(self, r):
        """
        Return the number of samples with count r.

        :param r: The amount of frequency.
        :type r: int
        :rtype: float
        """
        return math.exp(self._intercept + self._slope * math.log(r))

    def prob(self, sample):
        """
        Return the sample's probability.

        :param sample: sample of the event
        :type sample: str
        :rtype: float
        """
        count = self._freqdist[sample]
        p = self._prob_measure(count)
        if count == 0:
            if self._bins == self._freqdist.B():
                p = 0.0
            else:
                p = p / (self._bins - self._freqdist.B())
        else:
            p = p * self._renormal
        return p

    def _prob_measure(self, count):
        if count == 0 and self._freqdist.N() == 0:
            return 1.0
        elif count == 0 and self._freqdist.N() != 0:
            return self._freqdist.Nr(1) / self._freqdist.N()
        if self._switch_at > count:
            Er_1 = self._freqdist.Nr(count + 1)
            Er = self._freqdist.Nr(count)
        else:
            Er_1 = self.smoothedNr(count + 1)
            Er = self.smoothedNr(count)
        r_star = (count + 1) * Er_1 / Er
        return r_star / self._freqdist.N()

    def check(self):
        prob_sum = 0.0
        for i in range(0, len(self._Nr)):
            prob_sum += self._Nr[i] * self._prob_measure(i) / self._renormal
        print('Probability Sum:', prob_sum)

    def discount(self):
        """
        This function returns the total mass of probability transfers from the
        seen samples to the unseen samples.
        """
        return self.smoothedNr(1) / self._freqdist.N()

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def freqdist(self):
        return self._freqdist

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return '<SimpleGoodTuringProbDist based on %d samples>' % self._freqdist.N()