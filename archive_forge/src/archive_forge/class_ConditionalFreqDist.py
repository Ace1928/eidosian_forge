import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class ConditionalFreqDist(defaultdict):
    """
    A collection of frequency distributions for a single experiment
    run under different conditions.  Conditional frequency
    distributions are used to record the number of times each sample
    occurred, given the condition under which the experiment was run.
    For example, a conditional frequency distribution could be used to
    record the frequency of each word (type) in a document, given its
    length.  Formally, a conditional frequency distribution can be
    defined as a function that maps from each condition to the
    FreqDist for the experiment under that condition.

    Conditional frequency distributions are typically constructed by
    repeatedly running an experiment under a variety of conditions,
    and incrementing the sample outcome counts for the appropriate
    conditions.  For example, the following code will produce a
    conditional frequency distribution that encodes how often each
    word type occurs, given the length of that word type:

        >>> from nltk.probability import ConditionalFreqDist
        >>> from nltk.tokenize import word_tokenize
        >>> sent = "the the the dog dog some other words that we do not care about"
        >>> cfdist = ConditionalFreqDist()
        >>> for word in word_tokenize(sent):
        ...     condition = len(word)
        ...     cfdist[condition][word] += 1

    An equivalent way to do this is with the initializer:

        >>> cfdist = ConditionalFreqDist((len(word), word) for word in word_tokenize(sent))

    The frequency distribution for each condition is accessed using
    the indexing operator:

        >>> cfdist[3]
        FreqDist({'the': 3, 'dog': 2, 'not': 1})
        >>> cfdist[3].freq('the')
        0.5
        >>> cfdist[3]['dog']
        2

    When the indexing operator is used to access the frequency
    distribution for a condition that has not been accessed before,
    ``ConditionalFreqDist`` creates a new empty FreqDist for that
    condition.

    """

    def __init__(self, cond_samples=None):
        """
        Construct a new empty conditional frequency distribution.  In
        particular, the count for every sample, under every condition,
        is zero.

        :param cond_samples: The samples to initialize the conditional
            frequency distribution with
        :type cond_samples: Sequence of (condition, sample) tuples
        """
        defaultdict.__init__(self, FreqDist)
        if cond_samples:
            for cond, sample in cond_samples:
                self[cond][sample] += 1

    def __reduce__(self):
        kv_pairs = ((cond, self[cond]) for cond in self.conditions())
        return (self.__class__, (), None, None, kv_pairs)

    def conditions(self):
        """
        Return a list of the conditions that have been accessed for
        this ``ConditionalFreqDist``.  Use the indexing operator to
        access the frequency distribution for a given condition.
        Note that the frequency distributions for some conditions
        may contain zero sample outcomes.

        :rtype: list
        """
        return list(self.keys())

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this ``ConditionalFreqDist``.

        :rtype: int
        """
        return sum((fdist.N() for fdist in self.values()))

    def plot(self, *args, samples=None, title='', cumulative=False, percents=False, conditions=None, show=True, **kwargs):
        """
        Plot the given samples from the conditional frequency distribution.
        For a cumulative plot, specify cumulative=True. Additional ``*args`` and
        ``**kwargs`` are passed to matplotlib's plot function.
        (Requires Matplotlib to be installed.)

        :param samples: The samples to plot
        :type samples: list
        :param title: The title for the graph
        :type title: str
        :param cumulative: Whether the plot is cumulative. (default = False)
        :type cumulative: bool
        :param percents: Whether the plot uses percents instead of counts. (default = False)
        :type percents: bool
        :param conditions: The conditions to plot (default is all)
        :type conditions: list
        :param show: Whether to show the plot, or only return the ax.
        :type show: bool
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ValueError('The plot function requires matplotlib to be installed.See https://matplotlib.org/') from e
        if not conditions:
            conditions = self.conditions()
        else:
            conditions = [c for c in conditions if c in self]
        if not samples:
            samples = sorted({v for c in conditions for v in self[c]})
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 2
        ax = plt.gca()
        if conditions:
            freqs = []
            for condition in conditions:
                if cumulative:
                    freq = list(self[condition]._cumulative_frequencies(samples))
                else:
                    freq = [self[condition][sample] for sample in samples]
                if percents:
                    freq = [f / self[condition].N() * 100 for f in freq]
                freqs.append(freq)
            if cumulative:
                ylabel = 'Cumulative '
                legend_loc = 'lower right'
            else:
                ylabel = ''
                legend_loc = 'upper right'
            if percents:
                ylabel += 'Percents'
            else:
                ylabel += 'Counts'
            i = 0
            for freq in freqs:
                kwargs['label'] = conditions[i]
                i += 1
                ax.plot(freq, *args, **kwargs)
            ax.legend(loc=legend_loc)
            ax.grid(True, color='silver')
            ax.set_xticks(range(len(samples)))
            ax.set_xticklabels([str(s) for s in samples], rotation=90)
            if title:
                ax.set_title(title)
            ax.set_xlabel('Samples')
            ax.set_ylabel(ylabel)
        if show:
            plt.show()
        return ax

    def tabulate(self, *args, **kwargs):
        """
        Tabulate the given samples from the conditional frequency distribution.

        :param samples: The samples to plot
        :type samples: list
        :param conditions: The conditions to plot (default is all)
        :type conditions: list
        :param cumulative: A flag to specify whether the freqs are cumulative (default = False)
        :type title: bool
        """
        cumulative = _get_kwarg(kwargs, 'cumulative', False)
        conditions = _get_kwarg(kwargs, 'conditions', sorted(self.conditions()))
        samples = _get_kwarg(kwargs, 'samples', sorted({v for c in conditions if c in self for v in self[c]}))
        width = max((len('%s' % s) for s in samples))
        freqs = dict()
        for c in conditions:
            if cumulative:
                freqs[c] = list(self[c]._cumulative_frequencies(samples))
            else:
                freqs[c] = [self[c][sample] for sample in samples]
            width = max(width, max((len('%d' % f) for f in freqs[c])))
        condition_size = max((len('%s' % c) for c in conditions))
        print(' ' * condition_size, end=' ')
        for s in samples:
            print('%*s' % (width, s), end=' ')
        print()
        for c in conditions:
            print('%*s' % (condition_size, c), end=' ')
            for f in freqs[c]:
                print('%*d' % (width, f), end=' ')
            print()

    def __add__(self, other):
        """
        Add counts from two ConditionalFreqDists.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] += other[cond]
        return result

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] -= other[cond]
            if not result[cond]:
                del result[cond]
        return result

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = self.copy()
        for cond in other.conditions():
            result[cond] |= other[cond]
        return result

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] & other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        return result

    def __le__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types('<=', self, other)
        return set(self.conditions()).issubset(other.conditions()) and all((self[c] <= other[c] for c in self.conditions()))

    def __lt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types('<', self, other)
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types('>=', self, other)
        return other <= self

    def __gt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            raise_unorderable_types('>', self, other)
        return other < self

    def deepcopy(self):
        from copy import deepcopy
        return deepcopy(self)
    copy = deepcopy

    def __repr__(self):
        """
        Return a string representation of this ``ConditionalFreqDist``.

        :rtype: str
        """
        return '<ConditionalFreqDist with %d conditions>' % len(self)