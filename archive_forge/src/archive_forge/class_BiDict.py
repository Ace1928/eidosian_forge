from .links_base import Strand, Crossing, Link
import random
import collections
class BiDict:
    """
    A bijective mapping from range(n) to a set of hashable non-integers.

    >>> bd = BiDict({0:'a', 1:'b', 2:'c', 3:'d'})
    >>> bd[0], bd['b']
    ('a', 1)
    >>> bd._check()
    True
    >>> bd.insert_space(4, 2)
    >>> bd._check()
    False
    >>> bd[4] = 'e'
    >>> bd['f'] = 5
    >>> bd._check()
    True
    >>> bd
    BiDict({0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'})
    >>> bd.insert_space(3, 2)
    >>> bd[3], bd['y'] = 'x', 4
    >>> bd
    BiDict({0: 'a', 1: 'b', 2: 'c', 3: 'x', 4: 'y', 5: 'd', 6: 'e', 7: 'f'})
    >>> bd.pop(1); bd.pop('c'); bd.pop('e'); bd
    BiDict({0: 'a', 1: 'x', 2: 'y', 3: 'd', 4: 'f'})
    >>> (0 in bd, 'x' in bd, 'z' in bd, 10 in bd)
    (True, True, False, False)
    >>> bd.insert_space(-1, 2); bd[0] = 'u'; bd[1] = 'v'; bd
    BiDict({0: 'u', 1: 'v', 2: 'a', 3: 'x', 4: 'y', 5: 'd', 6: 'f'})
    >>> sorted(bd.values())
    ['a', 'd', 'f', 'u', 'v', 'x', 'y']
    """

    def __init__(self, int_to_set_dict):
        self.n = n = len(int_to_set_dict)
        assert sorted(int_to_set_dict) == list(range(n))
        self.int_to_set = int_to_set_dict
        self.set_to_int = {v: k for k, v in int_to_set_dict.items()}

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.int_to_set[index]
        elif index in self.set_to_int:
            return self.set_to_int[index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            assert not isinstance(value, int)
            the_int, the_set = (index, value)
        else:
            assert isinstance(value, int)
            the_int, the_set = (value, index)
        assert 0 <= the_int < self.n
        self.int_to_set[the_int] = the_set
        self.set_to_int[the_set] = the_int

    def __contains__(self, value):
        return value in self.set_to_int or value in self.int_to_set

    def pop(self, index):
        """
        Remove the given item and shift the indices downwards when necessary
        to close the gap.
        """
        if isinstance(index, int):
            the_int = index
            the_set = self.int_to_set[the_int]
        else:
            the_set = index
            the_int = self.set_to_int[the_set]
        self.int_to_set.pop(the_int)
        self.set_to_int.pop(the_set)

        def shift(j):
            return j if j < the_int else j - 1
        self.int_to_set = {shift(k): v for k, v in self.int_to_set.items()}
        self.set_to_int = {k: shift(v) for k, v in self.set_to_int.items()}
        self.n += -1

    def insert_space(self, i, n):
        """
        Shift indices upwards when necessary so that the n slots

        i, i + 1, ... , i + n - 1

        are unassigned.
        """
        assert isinstance(i, int)

        def shift(j):
            return j if j < i else j + n
        self.n = self.n + n
        self.int_to_set = {shift(k): v for k, v in self.int_to_set.items()}
        self.set_to_int = {k: shift(v) for k, v in self.set_to_int.items()}

    def values(self):
        return self.int_to_set.values()

    def __repr__(self):
        itos = self.int_to_set
        keys = sorted(itos)
        items = ['%d: %s' % (k, repr(itos[k])) for k in keys]
        return 'BiDict({' + ', '.join(items) + '})'

    def __len__(self):
        return len(self.int_to_set)

    def _check(self):
        return sorted(self.int_to_set) == list(range(self.n))