from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
class IterContainer(object):

    def __contains__(self, item):
        for o in self:
            if o == item:
                return True
        return False

    def __len__(self):
        return sum((1 for _ in self))

    def __getitem__(self, item):
        if isinstance(item, int):
            try:
                return next(islice(self, item, item + 1))
            except StopIteration:
                raise IndexError('index out of range')
        elif isinstance(item, slice):
            return islice(self, item.start, item.stop, item.step)

    def __iter__(self):
        raise NotImplementedError

    def index(self, item):
        for i, o in enumerate(self):
            if o == item:
                return i
        raise ValueError('%s is not in container' % item)

    def min(self, **kwargs):
        return min(self, **kwargs)

    def max(self, **kwargs):
        return max(self, **kwargs)

    def len(self):
        return len(self)

    def set(self):
        return set(self)

    def frozenset(self):
        return frozenset(self)

    def list(self):
        return list(iter(self))

    def tuple(self):
        return tuple(iter(self))

    def dict(self, **kwargs):
        return dict(self, **kwargs)

    def enumerate(self, start=0):
        return enumerate(self, start)

    def filter(self, function):
        return filter(function, self)

    def map(self, function):
        return map(function, self)

    def reduce(self, function, **kwargs):
        return reduce(function, self, **kwargs)

    def sum(self, *args, **kwargs):
        return sum(self, *args, **kwargs)

    def all(self):
        return all(self)

    def any(self):
        return any(self)

    def apply(self, function):
        for item in self:
            function(item)

    def counter(self):
        return Counter(self)

    def ordereddict(self):
        return OrderedDict(self)

    def cycle(self):
        return cycle(self)

    def chain(self, *others):
        return chain(self, *others)

    def dropwhile(self, predicate):
        return dropwhile(predicate, self)

    def takewhile(self, predicate):
        return takewhile(predicate, self)

    def ifilter(self, predicate):
        return ifilter(predicate, self)

    def ifilterfalse(self, predicate):
        return ifilterfalse(predicate, self)

    def imap(self, function):
        return imap(function, self)

    def starmap(self, function):
        return starmap(function, self)

    def islice(self, *args):
        return islice(self, *args)

    def compress(self, selectors):
        return compress(self, selectors)

    def groupby(self, *args, **kwargs):
        return groupby(self, *args, **kwargs)

    def tee(self, *args, **kwargs):
        return tee(self, *args, **kwargs)

    def permutations(self, *args, **kwargs):
        return permutations(self, *args, **kwargs)

    def combinations(self, *args, **kwargs):
        return combinations(self, *args, **kwargs)

    def combinations_with_replacement(self, *args, **kwargs):
        return combinations_with_replacement(self, *args, **kwargs)

    def izip(self, *args, **kwargs):
        return izip(self, *args, **kwargs)

    def izip_longest(self, *args, **kwargs):
        return izip_longest(self, *args, **kwargs)

    def product(self, *args, **kwargs):
        return product(self, *args, **kwargs)

    def __add__(self, other):
        return chain(self, other)

    def __iadd__(self, other):
        return chain(self, other)