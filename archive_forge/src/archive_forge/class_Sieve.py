import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
class Sieve:
    """An infinite list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('l', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    def __init__(self):
        self._n = 6
        self._list = _aset(2, 3, 5, 7, 11, 13)
        self._tlist = _aset(0, 1, 1, 2, 2, 4)
        self._mlist = _aset(0, 1, -1, -1, 0, -1)
        assert all((len(i) == self._n for i in (self._list, self._tlist, self._mlist)))

    def __repr__(self):
        return '<%s sieve (%i): %i, %i, %i, ... %i, %i\n%s sieve (%i): %i, %i, %i, ... %i, %i\n%s sieve (%i): %i, %i, %i, ... %i, %i>' % ('prime', len(self._list), self._list[0], self._list[1], self._list[2], self._list[-2], self._list[-1], 'totient', len(self._tlist), self._tlist[0], self._tlist[1], self._tlist[2], self._tlist[-2], self._tlist[-1], 'mobius', len(self._mlist), self._mlist[0], self._mlist[1], self._mlist[2], self._mlist[-2], self._mlist[-1])

    def _reset(self, prime=None, totient=None, mobius=None):
        """Reset all caches (default). To reset one or more set the
            desired keyword to True."""
        if all((i is None for i in (prime, totient, mobius))):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[:self._n]
        if totient:
            self._tlist = self._tlist[:self._n]
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        """Grow the sieve to cover all primes <= n (a real number).

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        if n <= self._list[-1]:
            return
        maxbase = int(n ** 0.5) + 1
        self.extend(maxbase)
        begin = self._list[-1] + 1
        newsieve = _arange(begin, n + 1)
        for p in self.primerange(maxbase):
            startindex = -begin % p
            for i in range(startindex, len(newsieve), p):
                newsieve[i] = 0
        self._list += _array('l', [x for x in newsieve if x])

    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('l', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = as_int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = _as_int_ceiling(a)
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))
            b = _as_int_ceiling(b)
        if a >= b:
            return
        self.extend(b)
        i = self.search(a)[1]
        maxi = len(self._list) + 1
        while i < maxi:
            p = self._list[i - 1]
            if p < b:
                yield p
                i += 1
            else:
                return

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += _arange(n, b)
            for i in range(1, n):
                ti = self._tlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._tlist[j] -= ti
                if i >= a:
                    yield ti
            for i in range(n, b):
                ti = self._tlist[i]
                for j in range(2 * i, b, i):
                    self._tlist[j] -= ti
                if i >= a:
                    yield ti

    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += _azeros(b - n)
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi
            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError('n should be >= 2 but got: %s' % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return (b, b)
        else:
            return (b, b + 1)

    def __contains__(self, n):
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        for n in count(1):
            yield self[n]

    def __getitem__(self, n):
        """Return the nth prime number"""
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                raise IndexError('Sieve indices start at 1.')
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            if n < 1:
                raise IndexError('Sieve indices start at 1.')
            n = as_int(n)
            self.extend_to_no(n)
            return self._list[n - 1]