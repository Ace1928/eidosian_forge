import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
class _BitString(str):
    """Helper class for binary string data (PRIVATE).

    Assistant class of binary string data used for storing and
    counting compatible clades in consensus tree searching. It includes
    some binary manipulation(&|^~) methods.

    _BitString is a sub-class of ``str`` object that only accepts two
    characters('0' and '1'), with additional functions for binary-like
    manipulation(&|^~). It is used to count and store the clades in
    multiple trees in consensus tree searching. During counting, the
    clades will be considered the same if their terminals(in terms of
    ``name`` attribute) are the same.

    For example, let's say two trees are provided as below to search
    their strict consensus tree::

        tree1: (((A, B), C),(D, E))
        tree2: ((A, (B, C)),(D, E))

    For both trees, a _BitString object '11111' will represent their
    root clade. Each '1' stands for the terminal clade in the list
    [A, B, C, D, E](the order might not be the same, it's determined
    by the ``get_terminal`` method of the first tree provided). For
    the clade ((A, B), C) in tree1 and (A, (B, C)) in tree2, they both
    can be represented by '11100'. Similarly, '11000' represents clade
    (A, B) in tree1, '01100' represents clade (B, C) in tree2, and '00011'
    represents clade (D, E) in both trees.

    So, with the ``_count_clades`` function in this module, finally we
    can get the clade counts and their _BitString representation as follows
    (the root and terminals are omitted)::

        clade   _BitString   count
        ABC     '11100'     2
        DE      '00011'     2
        AB      '11000'     1
        BC      '01100'     1

    To get the _BitString representation of a clade, we can use the following
    code snippet::

        # suppose we are provided with a tree list, the first thing to do is
        # to get all the terminal names in the first tree
        term_names = [term.name for term in trees[0].get_terminals()]
        # for a specific clade in any of the tree, also get its terminal names
        clade_term_names = [term.name for term in clade.get_terminals()]
        # then create a boolean list
        boolvals = [name in clade_term_names for name in term_names]
        # create the string version and pass it to _BitString
        bitstr = _BitString(''.join(map(str, map(int, boolvals))))
        # or, equivalently:
        bitstr = _BitString.from_bool(boolvals)

    To convert back::

        # get all the terminal clades of the first tree
        terms = [term for term in trees[0].get_terminals()]
        # get the index of terminal clades in bitstr
        index_list = bitstr.index_one()
        # get all terminal clades by index
        clade_terms = [terms[i] for i in index_list]
        # create a new calde and append all the terminal clades
        new_clade = BaseTree.Clade()
        new_clade.clades.extend(clade_terms)

    Examples
    --------
    >>> from Bio.Phylo.Consensus import _BitString
    >>> bitstr1 = _BitString('11111')
    >>> bitstr2 = _BitString('11100')
    >>> bitstr3 = _BitString('01101')
    >>> bitstr1
    _BitString('11111')
    >>> bitstr2 & bitstr3
    _BitString('01100')
    >>> bitstr2 | bitstr3
    _BitString('11101')
    >>> bitstr2 ^ bitstr3
    _BitString('10001')
    >>> bitstr2.index_one()
    [0, 1, 2]
    >>> bitstr3.index_one()
    [1, 2, 4]
    >>> bitstr3.index_zero()
    [0, 3]
    >>> bitstr1.contains(bitstr2)
    True
    >>> bitstr2.contains(bitstr3)
    False
    >>> bitstr2.independent(bitstr3)
    False
    >>> bitstr1.iscompatible(bitstr2)
    True
    >>> bitstr2.iscompatible(bitstr3)
    False

    """

    def __new__(cls, strdata):
        """Init from a binary string data."""
        if isinstance(strdata, str) and len(strdata) == strdata.count('0') + strdata.count('1'):
            return str.__new__(cls, strdata)
        else:
            raise TypeError("The input should be a binary string composed of '0' and '1'")

    def __and__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = selfint & otherint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __or__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = selfint | otherint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __xor__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = selfint ^ otherint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __rand__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = otherint & selfint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __ror__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = otherint | selfint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __rxor__(self, other):
        selfint = literal_eval('0b' + self)
        otherint = literal_eval('0b' + other)
        resultint = otherint ^ selfint
        return _BitString(bin(resultint)[2:].zfill(len(self)))

    def __repr__(self):
        return '_BitString(' + str.__repr__(self) + ')'

    def index_one(self):
        """Return a list of positions where the element is '1'."""
        return [i for i, n in enumerate(self) if n == '1']

    def index_zero(self):
        """Return a list of positions where the element is '0'."""
        return [i for i, n in enumerate(self) if n == '0']

    def contains(self, other):
        """Check if current bitstr1 contains another one bitstr2.

        That is to say, the bitstr2.index_one() is a subset of
        bitstr1.index_one().

        Examples:
            "011011" contains "011000", "011001", "000011"

        Be careful, "011011" also contains "000000". Actually, all _BitString
        objects contain all-zero _BitString of the same length.

        """
        xorbit = self ^ other
        return xorbit.count('1') == self.count('1') - other.count('1')

    def independent(self, other):
        """Check if current bitstr1 is independent of another one bitstr2.

        That is to say the bitstr1.index_one() and bitstr2.index_one() have
        no intersection.

        Be careful, all _BitString objects are independent of all-zero _BitString
        of the same length.
        """
        xorbit = self ^ other
        return xorbit.count('1') == self.count('1') + other.count('1')

    def iscompatible(self, other):
        """Check if current bitstr1 is compatible with another bitstr2.

        Two conditions are considered as compatible:
         1. bitstr1.contain(bitstr2) or vice versa;
         2. bitstr1.independent(bitstr2).

        """
        return self.contains(other) or other.contains(self) or self.independent(other)

    @classmethod
    def from_bool(cls, bools):
        return cls(''.join(map(str, map(int, bools))))