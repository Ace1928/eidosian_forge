from operator import inv
class Perm4:
    """
    Class Perm4: A permutation of {0,1,2,3}.

    A permutation can be initialized with a length 4 dictionary or a
    4-tuple or a length 2 dictionary; in the latter case the sign of
    the permutation can be specified by setting "sign=0" (even) or
    "sign=1" (odd).  The default sign is odd, since odd permutations
    describe orientation-preserving gluings.
    """

    def __init__(self, init, sign=1):
        if isinstance(init, int):
            self._index = init
            self._tuple = S4_tuples[init]
        elif isinstance(init, Perm4):
            self._index = init._index
            self._tuple = init._tuple
        elif len(init) == 4:
            self._tuple = tuple((init[i] for i in range(4)))
            self._index = perm_tuple_to_index[self._tuple]
        else:
            self._tuple = Perm4Basic(init, sign).tuple()
            self._index = perm_tuple_to_index[self._tuple]

    def image(self, bitmap):
        """
        A subset of {0,1,2,3} can be represented by a bitmap.  This
        computes the bitmap of the image subset.

        >>> Perm4([2, 3, 1, 0]).image(10)
        9
        """
        return bitmap_images[self._index, bitmap]

    def __repr__(self):
        return str(self._tuple)

    def __call__(self, a_tuple):
        """
        P((i, ... ,j)) returns the image tuple (P(i), ... , P(j))

        >>> Perm4([2, 3, 1, 0])(range(3))
        (2, 3, 1)
        """
        image = []
        for i in a_tuple:
            image.append(self._tuple[i])
        return tuple(image)

    def __getitem__(self, index):
        """
        P[i] returns the image of i, P(i)

        >>> Perm4([2, 3, 1, 0])[3]
        0
        """
        return self._tuple[index]

    def __mul__(self, other):
        """
        P*Q is the composition P*Q(i) = P(Q(i))

        >>> P = Perm4((2, 3, 1, 0))
        >>> Q = Perm4((1, 0, 2, 3))
        >>> P * Q
        (3, 2, 1, 0)
        >>> Q * P
        (2, 3, 0, 1)
        """
        return mult_table_by_index[self._index, other._index]

    def __invert__(self):
        """
        inv(P) is the inverse permutation

        >>> inv(Perm4([2, 1, 3, 0]))
        (3, 1, 0, 2)
        """
        return inverse_by_index[self._index]

    def sign(self):
        """
        sign(P) is the sign: 0 for even, 1 for odd

        >>> Perm4([0, 1, 3, 2]).sign()
        1
        >>> Perm4([1, 0, 3, 2]).sign()
        0
        """
        return perm_signs_by_index[self._index]

    def tuple(self):
        """
        P.tuple() returns a tuple representing the permutation P

        >>> Perm4([1, 2, 0, 3]).tuple()
        (1, 2, 0, 3)
        """
        return self._tuple

    @staticmethod
    def S4():
        """"
        All permutations in S4

        >>> len(list(Perm4.S4()))
        24
        """
        for p in S4_tuples:
            yield Perm4(p)

    @staticmethod
    def A4():
        """
        All even permutations in A4

        >>> len(list(Perm4.A4()))
        12
        """
        for p in A4_tuples:
            yield Perm4(p)

    @staticmethod
    def KleinFour():
        """
        Z/2 x Z/2 as a subgroup of A4.

        >>> len(list(Perm4.KleinFour()))
        4
        """
        for p in KleinFour_tuples:
            yield Perm4(p)