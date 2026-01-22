import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
class _SeqAbstractBaseClass(ABC):
    """Abstract base class for the Seq and MutableSeq classes (PRIVATE).

    Most users will not need to use this class. It is used internally as an
    abstract base class for Seq and MutableSeq, as most of their methods are
    identical.
    """
    __slots__ = ('_data',)
    __array_ufunc__ = None

    @abstractmethod
    def __init__(self):
        pass

    def __bytes__(self):
        return bytes(self._data)

    def __repr__(self):
        """Return (truncated) representation of the sequence."""
        data = self._data
        if isinstance(data, _UndefinedSequenceData):
            return f'Seq(None, length={len(self)})'
        if isinstance(data, _PartiallyDefinedSequenceData):
            d = {}
            for position, seq in data._data.items():
                if len(seq) > 60:
                    start = seq[:54].decode('ASCII')
                    end = seq[-3:].decode('ASCII')
                    seq = f'{start}...{end}'
                else:
                    seq = seq.decode('ASCII')
                d[position] = seq
            return 'Seq(%r, length=%d)' % (d, len(self))
        if len(data) > 60:
            start = data[:54].decode('ASCII')
            end = data[-3:].decode('ASCII')
            return f"{self.__class__.__name__}('{start}...{end}')"
        else:
            data = data.decode('ASCII')
            return f"{self.__class__.__name__}('{data}')"

    def __str__(self):
        """Return the full sequence as a python string."""
        return self._data.decode('ASCII')

    def __eq__(self, other):
        """Compare the sequence to another sequence or a string.

        Sequences are equal to each other if their sequence contents is
        identical:

        >>> from Bio.Seq import Seq, MutableSeq
        >>> seq1 = Seq("ACGT")
        >>> seq2 = Seq("ACGT")
        >>> mutable_seq = MutableSeq("ACGT")
        >>> seq1 == seq2
        True
        >>> seq1 == mutable_seq
        True
        >>> seq1 == "ACGT"
        True

        Note that the sequence objects themselves are not identical to each
        other:

        >>> id(seq1) == id(seq2)
        False
        >>> seq1 is seq2
        False

        Sequences can also be compared to strings, ``bytes``, and ``bytearray``
        objects:

        >>> seq1 == "ACGT"
        True
        >>> seq1 == b"ACGT"
        True
        >>> seq1 == bytearray(b"ACGT")
        True
        """
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data == other._data
        elif isinstance(other, str):
            return self._data == other.encode('ASCII')
        else:
            return self._data == other

    def __lt__(self, other):
        """Implement the less-than operand."""
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data < other._data
        elif isinstance(other, str):
            return self._data < other.encode('ASCII')
        else:
            return self._data < other

    def __le__(self, other):
        """Implement the less-than or equal operand."""
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data <= other._data
        elif isinstance(other, str):
            return self._data <= other.encode('ASCII')
        else:
            return self._data <= other

    def __gt__(self, other):
        """Implement the greater-than operand."""
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data > other._data
        elif isinstance(other, str):
            return self._data > other.encode('ASCII')
        else:
            return self._data > other

    def __ge__(self, other):
        """Implement the greater-than or equal operand."""
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data >= other._data
        elif isinstance(other, str):
            return self._data >= other.encode('ASCII')
        else:
            return self._data >= other

    def __len__(self):
        """Return the length of the sequence."""
        return len(self._data)

    def __iter__(self):
        """Return an iterable of the sequence."""
        return self._data.decode('ASCII').__iter__()

    @overload
    def __getitem__(self, index: int) -> str:
        ...

    @overload
    def __getitem__(self, index: slice) -> 'Seq':
        ...

    def __getitem__(self, index):
        """Return a subsequence as a single letter or as a sequence object.

        If the index is an integer, a single letter is returned as a Python
        string:

        >>> seq = Seq('ACTCGACGTCG')
        >>> seq[5]
        'A'

        Otherwise, a new sequence object of the same class is returned:

        >>> seq[5:8]
        Seq('ACG')
        >>> mutable_seq = MutableSeq('ACTCGACGTCG')
        >>> mutable_seq[5:8]
        MutableSeq('ACG')
        """
        if isinstance(index, numbers.Integral):
            return chr(self._data[index])
        else:
            return self.__class__(self._data[index])

    def __add__(self, other):
        """Add a sequence or string to this sequence.

        >>> from Bio.Seq import Seq, MutableSeq
        >>> Seq("MELKI") + "LV"
        Seq('MELKILV')
        >>> MutableSeq("MELKI") + "LV"
        MutableSeq('MELKILV')
        """
        if isinstance(other, _SeqAbstractBaseClass):
            return self.__class__(self._data + other._data)
        elif isinstance(other, str):
            return self.__class__(self._data + other.encode('ASCII'))
        else:
            return NotImplemented

    def __radd__(self, other):
        """Add a sequence string on the left.

        >>> from Bio.Seq import Seq, MutableSeq
        >>> "LV" + Seq("MELKI")
        Seq('LVMELKI')
        >>> "LV" + MutableSeq("MELKI")
        MutableSeq('LVMELKI')

        Adding two sequence objects is handled via the __add__ method.
        """
        if isinstance(other, str):
            return self.__class__(other.encode('ASCII') + self._data)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiply sequence by integer.

        >>> from Bio.Seq import Seq, MutableSeq
        >>> Seq('ATG') * 2
        Seq('ATGATG')
        >>> MutableSeq('ATG') * 2
        MutableSeq('ATGATG')
        """
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def __rmul__(self, other):
        """Multiply integer by sequence.

        >>> from Bio.Seq import Seq
        >>> 2 * Seq('ATG')
        Seq('ATGATG')
        """
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def __imul__(self, other):
        """Multiply the sequence object by other and assign.

        >>> from Bio.Seq import Seq
        >>> seq = Seq('ATG')
        >>> seq *= 2
        >>> seq
        Seq('ATGATG')

        Note that this is different from in-place multiplication. The ``seq``
        variable is reassigned to the multiplication result, but any variable
        pointing to ``seq`` will remain unchanged:

        >>> seq = Seq('ATG')
        >>> seq2 = seq
        >>> id(seq) == id(seq2)
        True
        >>> seq *= 2
        >>> seq
        Seq('ATGATG')
        >>> seq2
        Seq('ATG')
        >>> id(seq) == id(seq2)
        False
        """
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def count(self, sub, start=None, end=None):
        """Return a non-overlapping count, like that of a python string.

        The number of occurrences of substring argument sub in the
        (sub)sequence given by [start:end] is returned as an integer.
        Optional arguments start and end are interpreted as in slice
        notation.

        Arguments:
         - sub - a string or another Seq object to look for
         - start - optional integer, slice start
         - end - optional integer, slice end

        e.g.

        >>> from Bio.Seq import Seq
        >>> my_seq = Seq("AAAATGA")
        >>> print(my_seq.count("A"))
        5
        >>> print(my_seq.count("ATG"))
        1
        >>> print(my_seq.count(Seq("AT")))
        1
        >>> print(my_seq.count("AT", 2, -1))
        1

        HOWEVER, please note because the ``count`` method of Seq and MutableSeq
        objects, like that of Python strings, do a non-overlapping search, this
        may not give the answer you expect:

        >>> "AAAA".count("AA")
        2
        >>> print(Seq("AAAA").count("AA"))
        2

        For an overlapping search, use the ``count_overlap`` method:

        >>> print(Seq("AAAA").count_overlap("AA"))
        3
        """
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.count(sub, start, end)

    def count_overlap(self, sub, start=None, end=None):
        """Return an overlapping count.

        Returns an integer, the number of occurrences of substring
        argument sub in the (sub)sequence given by [start:end].
        Optional arguments start and end are interpreted as in slice
        notation.

        Arguments:
         - sub - a string or another Seq object to look for
         - start - optional integer, slice start
         - end - optional integer, slice end

        e.g.

        >>> from Bio.Seq import Seq
        >>> print(Seq("AAAA").count_overlap("AA"))
        3
        >>> print(Seq("ATATATATA").count_overlap("ATA"))
        4
        >>> print(Seq("ATATATATA").count_overlap("ATA", 3, -1))
        1

        For a non-overlapping search, use the ``count`` method:

        >>> print(Seq("AAAA").count("AA"))
        2

        Where substrings do not overlap, ``count_overlap`` behaves the same as
        the ``count`` method:

        >>> from Bio.Seq import Seq
        >>> my_seq = Seq("AAAATGA")
        >>> print(my_seq.count_overlap("A"))
        5
        >>> my_seq.count_overlap("A") == my_seq.count("A")
        True
        >>> print(my_seq.count_overlap("ATG"))
        1
        >>> my_seq.count_overlap("ATG") == my_seq.count("ATG")
        True
        >>> print(my_seq.count_overlap(Seq("AT")))
        1
        >>> my_seq.count_overlap(Seq("AT")) == my_seq.count(Seq("AT"))
        True
        >>> print(my_seq.count_overlap("AT", 2, -1))
        1
        >>> my_seq.count_overlap("AT", 2, -1) == my_seq.count("AT", 2, -1)
        True

        HOWEVER, do not use this method for such cases because the
        count() method is much for efficient.
        """
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        data = self._data
        overlap_count = 0
        while True:
            start = data.find(sub, start, end) + 1
            if start != 0:
                overlap_count += 1
            else:
                return overlap_count

    def __contains__(self, item):
        """Return True if item is a subsequence of the sequence, and False otherwise.

        e.g.

        >>> from Bio.Seq import Seq, MutableSeq
        >>> my_dna = Seq("ATATGAAATTTGAAAA")
        >>> "AAA" in my_dna
        True
        >>> Seq("AAA") in my_dna
        True
        >>> MutableSeq("AAA") in my_dna
        True
        """
        if isinstance(item, _SeqAbstractBaseClass):
            item = bytes(item)
        elif isinstance(item, str):
            item = item.encode('ASCII')
        return item in self._data

    def find(self, sub, start=None, end=None):
        """Return the lowest index in the sequence where subsequence sub is found.

        With optional arguments start and end, return the lowest index in the
        sequence such that the subsequence sub is contained within the sequence
        region [start:end].

        Arguments:
         - sub - a string or another Seq or MutableSeq object to search for
         - start - optional integer, slice start
         - end - optional integer, slice end

        Returns -1 if the subsequence is NOT found.

        e.g. Locating the first typical start codon, AUG, in an RNA sequence:

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.find("AUG")
        3

        The next typical start codon can then be found by starting the search
        at position 4:

        >>> my_rna.find("AUG", 4)
        15

        See the ``search`` method to find the locations of multiple subsequences
        at the same time.
        """
        if isinstance(sub, _SeqAbstractBaseClass):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.find(sub, start, end)

    def rfind(self, sub, start=None, end=None):
        """Return the highest index in the sequence where subsequence sub is found.

        With optional arguments start and end, return the highest index in the
        sequence such that the subsequence sub is contained within the sequence
        region [start:end].

        Arguments:
         - sub - a string or another Seq or MutableSeq object to search for
         - start - optional integer, slice start
         - end - optional integer, slice end

        Returns -1 if the subsequence is NOT found.

        e.g. Locating the last typical start codon, AUG, in an RNA sequence:

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.rfind("AUG")
        15

        The location of the typical start codon before that can be found by
        ending the search at position 15:

        >>> my_rna.rfind("AUG", end=15)
        3

        See the ``search`` method to find the locations of multiple subsequences
        at the same time.
        """
        if isinstance(sub, _SeqAbstractBaseClass):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.rfind(sub, start, end)

    def index(self, sub, start=None, end=None):
        """Return the lowest index in the sequence where subsequence sub is found.

        With optional arguments start and end, return the lowest index in the
        sequence such that the subsequence sub is contained within the sequence
        region [start:end].

        Arguments:
         - sub - a string or another Seq or MutableSeq object to search for
         - start - optional integer, slice start
         - end - optional integer, slice end

        Raises a ValueError if the subsequence is NOT found.

        e.g. Locating the first typical start codon, AUG, in an RNA sequence:

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.index("AUG")
        3

        The next typical start codon can then be found by starting the search
        at position 4:

        >>> my_rna.index("AUG", 4)
        15

        This method performs the same search as the ``find`` method.  However,
        if the subsequence is not found, ``find`` returns -1 while ``index``
        raises a ValueError:

        >>> my_rna.index("T")
        Traceback (most recent call last):
                   ...
        ValueError: ...
        >>> my_rna.find("T")
        -1

        See the ``search`` method to find the locations of multiple subsequences
        at the same time.
        """
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.index(sub, start, end)

    def rindex(self, sub, start=None, end=None):
        """Return the highest index in the sequence where subsequence sub is found.

        With optional arguments start and end, return the highest index in the
        sequence such that the subsequence sub is contained within the sequence
        region [start:end].

        Arguments:
         - sub - a string or another Seq or MutableSeq object to search for
         - start - optional integer, slice start
         - end - optional integer, slice end

        Returns -1 if the subsequence is NOT found.

        e.g. Locating the last typical start codon, AUG, in an RNA sequence:

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.rindex("AUG")
        15

        The location of the typical start codon before that can be found by
        ending the search at position 15:

        >>> my_rna.rindex("AUG", end=15)
        3

        This method performs the same search as the ``rfind`` method.  However,
        if the subsequence is not found, ``rfind`` returns -1 which ``rindex``
        raises a ValueError:

        >>> my_rna.rindex("T")
        Traceback (most recent call last):
                   ...
        ValueError: ...
        >>> my_rna.rfind("T")
        -1

        See the ``search`` method to find the locations of multiple subsequences
        at the same time.
        """
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.rindex(sub, start, end)

    def search(self, subs):
        """Search the substrings subs in self and yield the index and substring found.

        Arguments:
         - subs - a list of strings, Seq, MutableSeq, bytes, or bytearray
           objects containing the substrings to search for.

        >>> from Bio.Seq import Seq
        >>> dna = Seq("GTCATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTTG")
        >>> matches = dna.search(["CC", Seq("ATTG"), "ATTG", Seq("CCC")])
        >>> for index, substring in matches:
        ...     print(index, substring)
        ...
        7 CC
        9 ATTG
        20 CC
        34 CC
        34 CCC
        35 CC
        """
        subdict = collections.defaultdict(set)
        for index, sub in enumerate(subs):
            if isinstance(sub, (_SeqAbstractBaseClass, bytearray)):
                sub = bytes(sub)
            elif isinstance(sub, str):
                sub = sub.encode('ASCII')
            elif not isinstance(sub, bytes):
                raise TypeError("subs[%d]: a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % (index, type(sub)))
            length = len(sub)
            subdict[length].add(sub)
        for start in range(len(self) - 1):
            for length, subs in subdict.items():
                stop = start + length
                for sub in subs:
                    if self._data[start:stop] == sub:
                        yield (start, sub.decode())
                        break

    def startswith(self, prefix, start=None, end=None):
        """Return True if the sequence starts with the given prefix, False otherwise.

        Return True if the sequence starts with the specified prefix
        (a string or another Seq object), False otherwise.
        With optional start, test sequence beginning at that position.
        With optional end, stop comparing sequence at that position.
        prefix can also be a tuple of strings to try.  e.g.

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.startswith("GUC")
        True
        >>> my_rna.startswith("AUG")
        False
        >>> my_rna.startswith("AUG", 3)
        True
        >>> my_rna.startswith(("UCC", "UCA", "UCG"), 1)
        True
        """
        if isinstance(prefix, tuple):
            prefix = tuple((bytes(p) if isinstance(p, _SeqAbstractBaseClass) else p.encode('ASCII') for p in prefix))
        elif isinstance(prefix, _SeqAbstractBaseClass):
            prefix = bytes(prefix)
        elif isinstance(prefix, str):
            prefix = prefix.encode('ASCII')
        return self._data.startswith(prefix, start, end)

    def endswith(self, suffix, start=None, end=None):
        """Return True if the sequence ends with the given suffix, False otherwise.

        Return True if the sequence ends with the specified suffix
        (a string or another Seq object), False otherwise.
        With optional start, test sequence beginning at that position.
        With optional end, stop comparing sequence at that position.
        suffix can also be a tuple of strings to try.  e.g.

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_rna.endswith("UUG")
        True
        >>> my_rna.endswith("AUG")
        False
        >>> my_rna.endswith("AUG", 0, 18)
        True
        >>> my_rna.endswith(("UCC", "UCA", "UUG"))
        True
        """
        if isinstance(suffix, tuple):
            suffix = tuple((bytes(p) if isinstance(p, _SeqAbstractBaseClass) else p.encode('ASCII') for p in suffix))
        elif isinstance(suffix, _SeqAbstractBaseClass):
            suffix = bytes(suffix)
        elif isinstance(suffix, str):
            suffix = suffix.encode('ASCII')
        return self._data.endswith(suffix, start, end)

    def split(self, sep=None, maxsplit=-1):
        """Return a list of subsequences when splitting the sequence by separator sep.

        Return a list of the subsequences in the sequence (as Seq objects),
        using sep as the delimiter string.  If maxsplit is given, at
        most maxsplit splits are done.  If maxsplit is omitted, all
        splits are made.

        For consistency with the ``split`` method of Python strings, any
        whitespace (tabs, spaces, newlines) is a separator if sep is None, the
        default value

        e.g.

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_aa = my_rna.translate()
        >>> my_aa
        Seq('VMAIVMGR*KGAR*L')
        >>> for pep in my_aa.split("*"):
        ...     pep
        Seq('VMAIVMGR')
        Seq('KGAR')
        Seq('L')
        >>> for pep in my_aa.split("*", 1):
        ...     pep
        Seq('VMAIVMGR')
        Seq('KGAR*L')

        See also the rsplit method, which splits the sequence starting from the
        end:

        >>> for pep in my_aa.rsplit("*", 1):
        ...     pep
        Seq('VMAIVMGR*KGAR')
        Seq('L')
        """
        if isinstance(sep, _SeqAbstractBaseClass):
            sep = bytes(sep)
        elif isinstance(sep, str):
            sep = sep.encode('ASCII')
        return [Seq(part) for part in self._data.split(sep, maxsplit)]

    def rsplit(self, sep=None, maxsplit=-1):
        """Return a list of subsequences by splitting the sequence from the right.

        Return a list of the subsequences in the sequence (as Seq objects),
        using sep as the delimiter string.  If maxsplit is given, at
        most maxsplit splits are done.  If maxsplit is omitted, all
        splits are made.

        For consistency with the ``rsplit`` method of Python strings, any
        whitespace (tabs, spaces, newlines) is a separator if sep is None, the
        default value

        e.g.

        >>> from Bio.Seq import Seq
        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")
        >>> my_aa = my_rna.translate()
        >>> my_aa
        Seq('VMAIVMGR*KGAR*L')
        >>> for pep in my_aa.rsplit("*"):
        ...     pep
        Seq('VMAIVMGR')
        Seq('KGAR')
        Seq('L')
        >>> for pep in my_aa.rsplit("*", 1):
        ...     pep
        Seq('VMAIVMGR*KGAR')
        Seq('L')

        See also the split method, which splits the sequence starting from the
        beginning:

        >>> for pep in my_aa.split("*", 1):
        ...     pep
        Seq('VMAIVMGR')
        Seq('KGAR*L')
        """
        if isinstance(sep, _SeqAbstractBaseClass):
            sep = bytes(sep)
        elif isinstance(sep, str):
            sep = sep.encode('ASCII')
        return [Seq(part) for part in self._data.rsplit(sep, maxsplit)]

    def strip(self, chars=None, inplace=False):
        """Return a sequence object with leading and trailing ends stripped.

        With default arguments, leading and trailing whitespace is removed:

        >>> seq = Seq(" ACGT ")
        >>> seq.strip()
        Seq('ACGT')
        >>> seq
        Seq(' ACGT ')

        If ``chars`` is given and not ``None``, remove characters in ``chars``
        instead.  The order of the characters to be removed is not important:

        >>> Seq("ACGTACGT").strip("TGCA")
        Seq('')

        A copy of the sequence is returned if ``inplace`` is ``False`` (the
        default value).  If ``inplace`` is ``True``, the sequence is stripped
        in-place and returned.

        >>> seq = MutableSeq(" ACGT ")
        >>> seq.strip()
        MutableSeq('ACGT')
        >>> seq
        MutableSeq(' ACGT ')
        >>> seq.strip(inplace=True)
        MutableSeq('ACGT')
        >>> seq
        MutableSeq('ACGT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if ``strip``
        is called on a ``Seq`` object with ``inplace=True``.

        See also the lstrip and rstrip methods.
        """
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.strip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def lstrip(self, chars=None, inplace=False):
        """Return a sequence object with leading and trailing ends stripped.

        With default arguments, leading whitespace is removed:

        >>> seq = Seq(" ACGT ")
        >>> seq.lstrip()
        Seq('ACGT ')
        >>> seq
        Seq(' ACGT ')

        If ``chars`` is given and not ``None``, remove characters in ``chars``
        from the leading end instead.  The order of the characters to be removed
        is not important:

        >>> Seq("ACGACGTTACG").lstrip("GCA")
        Seq('TTACG')

        A copy of the sequence is returned if ``inplace`` is ``False`` (the
        default value).  If ``inplace`` is ``True``, the sequence is stripped
        in-place and returned.

        >>> seq = MutableSeq(" ACGT ")
        >>> seq.lstrip()
        MutableSeq('ACGT ')
        >>> seq
        MutableSeq(' ACGT ')
        >>> seq.lstrip(inplace=True)
        MutableSeq('ACGT ')
        >>> seq
        MutableSeq('ACGT ')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``lstrip`` is called on a ``Seq`` object with ``inplace=True``.

        See also the strip and rstrip methods.
        """
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.lstrip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def rstrip(self, chars=None, inplace=False):
        """Return a sequence object with trailing ends stripped.

        With default arguments, trailing whitespace is removed:

        >>> seq = Seq(" ACGT ")
        >>> seq.rstrip()
        Seq(' ACGT')
        >>> seq
        Seq(' ACGT ')

        If ``chars`` is given and not ``None``, remove characters in ``chars``
        from the trailing end instead.  The order of the characters to be
        removed is not important:

        >>> Seq("ACGACGTTACG").rstrip("GCA")
        Seq('ACGACGTT')

        A copy of the sequence is returned if ``inplace`` is ``False`` (the
        default value).  If ``inplace`` is ``True``, the sequence is stripped
        in-place and returned.

        >>> seq = MutableSeq(" ACGT ")
        >>> seq.rstrip()
        MutableSeq(' ACGT')
        >>> seq
        MutableSeq(' ACGT ')
        >>> seq.rstrip(inplace=True)
        MutableSeq(' ACGT')
        >>> seq
        MutableSeq(' ACGT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``rstrip`` is called on a ``Seq`` object with ``inplace=True``.

        See also the strip and lstrip methods.
        """
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.rstrip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def removeprefix(self, prefix, inplace=False):
        """Return a new Seq object with prefix (left) removed.

        This behaves like the python string method of the same name.

        e.g. Removing a start Codon:

        >>> from Bio.Seq import Seq
        >>> my_seq = Seq("ATGGTGTGTGT")
        >>> my_seq
        Seq('ATGGTGTGTGT')
        >>> my_seq.removeprefix('ATG')
        Seq('GTGTGTGT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``removeprefix`` is called on a ``Seq`` object with ``inplace=True``.

        See also the removesuffix method.
        """
        if isinstance(prefix, _SeqAbstractBaseClass):
            prefix = bytes(prefix)
        elif isinstance(prefix, str):
            prefix = prefix.encode('ASCII')
        try:
            data = self._data.removeprefix(prefix)
        except TypeError:
            raise TypeError('argument must be a string, Seq, MutableSeq, or bytes-like object') from None
        except AttributeError:
            data = self._data
            if data.startswith(prefix):
                data = data[len(prefix):]
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def removesuffix(self, suffix, inplace=False):
        """Return a new Seq object with suffix (right) removed.

        This behaves like the python string method of the same name.

        e.g. Removing a stop codon:

        >>> from Bio.Seq import Seq
        >>> my_seq = Seq("GTGTGTGTTAG")
        >>> my_seq
        Seq('GTGTGTGTTAG')
        >>> stop_codon = Seq("TAG")
        >>> my_seq.removesuffix(stop_codon)
        Seq('GTGTGTGT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``removesuffix`` is called on a ``Seq`` object with ``inplace=True``.

        See also the removeprefix method.
        """
        if isinstance(suffix, _SeqAbstractBaseClass):
            suffix = bytes(suffix)
        elif isinstance(suffix, str):
            suffix = suffix.encode('ASCII')
        try:
            data = self._data.removesuffix(suffix)
        except TypeError:
            raise TypeError('argument must be a string, Seq, MutableSeq, or bytes-like object') from None
        except AttributeError:
            data = self._data
            if data.endswith(suffix):
                data = data[:-len(suffix)]
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def upper(self, inplace=False):
        """Return the sequence in upper case.

        An upper-case copy of the sequence is returned if inplace is False,
        the default value:

        >>> from Bio.Seq import Seq, MutableSeq
        >>> my_seq = Seq("VHLTPeeK*")
        >>> my_seq
        Seq('VHLTPeeK*')
        >>> my_seq.lower()
        Seq('vhltpeek*')
        >>> my_seq.upper()
        Seq('VHLTPEEK*')
        >>> my_seq
        Seq('VHLTPeeK*')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("VHLTPeeK*")
        >>> my_seq
        MutableSeq('VHLTPeeK*')
        >>> my_seq.lower()
        MutableSeq('vhltpeek*')
        >>> my_seq.upper()
        MutableSeq('VHLTPEEK*')
        >>> my_seq
        MutableSeq('VHLTPeeK*')

        >>> my_seq.lower(inplace=True)
        MutableSeq('vhltpeek*')
        >>> my_seq
        MutableSeq('vhltpeek*')
        >>> my_seq.upper(inplace=True)
        MutableSeq('VHLTPEEK*')
        >>> my_seq
        MutableSeq('VHLTPEEK*')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``upper`` is called on a ``Seq`` object with ``inplace=True``.

        See also the ``lower`` method.
        """
        data = self._data.upper()
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def lower(self, inplace=False):
        """Return the sequence in lower case.

        An lower-case copy of the sequence is returned if inplace is False,
        the default value:

        >>> from Bio.Seq import Seq, MutableSeq
        >>> my_seq = Seq("VHLTPeeK*")
        >>> my_seq
        Seq('VHLTPeeK*')
        >>> my_seq.lower()
        Seq('vhltpeek*')
        >>> my_seq.upper()
        Seq('VHLTPEEK*')
        >>> my_seq
        Seq('VHLTPeeK*')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("VHLTPeeK*")
        >>> my_seq
        MutableSeq('VHLTPeeK*')
        >>> my_seq.lower()
        MutableSeq('vhltpeek*')
        >>> my_seq.upper()
        MutableSeq('VHLTPEEK*')
        >>> my_seq
        MutableSeq('VHLTPeeK*')

        >>> my_seq.lower(inplace=True)
        MutableSeq('vhltpeek*')
        >>> my_seq
        MutableSeq('vhltpeek*')
        >>> my_seq.upper(inplace=True)
        MutableSeq('VHLTPEEK*')
        >>> my_seq
        MutableSeq('VHLTPEEK*')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``lower`` is called on a ``Seq`` object with ``inplace=True``.

        See also the ``upper`` method.
        """
        data = self._data.lower()
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def isupper(self):
        """Return True if all ASCII characters in data are uppercase.

        If there are no cased characters, the method returns False.
        """
        return self._data.isupper()

    def islower(self):
        """Return True if all ASCII characters in data are lowercase.

        If there are no cased characters, the method returns False.
        """
        return self._data.islower()

    def translate(self, table='Standard', stop_symbol='*', to_stop=False, cds=False, gap='-'):
        """Turn a nucleotide sequence into a protein sequence by creating a new sequence object.

        This method will translate DNA or RNA sequences. It should not
        be used on protein sequences as any result will be biologically
        meaningless.

        Arguments:
         - table - Which codon table to use?  This can be either a name
           (string), an NCBI identifier (integer), or a CodonTable
           object (useful for non-standard genetic codes).  This
           defaults to the "Standard" table.
         - stop_symbol - Single character string, what to use for
           terminators.  This defaults to the asterisk, "*".
         - to_stop - Boolean, defaults to False meaning do a full
           translation continuing on past any stop codons (translated as the
           specified stop_symbol).  If True, translation is terminated at
           the first in frame stop codon (and the stop_symbol is not
           appended to the returned protein sequence).
         - cds - Boolean, indicates this is a complete CDS.  If True,
           this checks the sequence starts with a valid alternative start
           codon (which will be translated as methionine, M), that the
           sequence length is a multiple of three, and that there is a
           single in frame stop codon at the end (this will be excluded
           from the protein sequence, regardless of the to_stop option).
           If these tests fail, an exception is raised.
         - gap - Single character string to denote symbol used for gaps.
           Defaults to the minus sign.

        A ``Seq`` object is returned if ``translate`` is called on a ``Seq``
        object; a ``MutableSeq`` object is returned if ``translate`` is called
        pn a ``MutableSeq`` object.

        e.g. Using the standard table:

        >>> coding_dna = Seq("GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
        >>> coding_dna.translate()
        Seq('VAIVMGR*KGAR*')
        >>> coding_dna.translate(stop_symbol="@")
        Seq('VAIVMGR@KGAR@')
        >>> coding_dna.translate(to_stop=True)
        Seq('VAIVMGR')

        Now using NCBI table 2, where TGA is not a stop codon:

        >>> coding_dna.translate(table=2)
        Seq('VAIVMGRWKGAR*')
        >>> coding_dna.translate(table=2, to_stop=True)
        Seq('VAIVMGRWKGAR')

        In fact, GTG is an alternative start codon under NCBI table 2, meaning
        this sequence could be a complete CDS:

        >>> coding_dna.translate(table=2, cds=True)
        Seq('MAIVMGRWKGAR')

        It isn't a valid CDS under NCBI table 1, due to both the start codon
        and also the in frame stop codons:

        >>> coding_dna.translate(table=1, cds=True)
        Traceback (most recent call last):
            ...
        Bio.Data.CodonTable.TranslationError: First codon 'GTG' is not a start codon

        If the sequence has no in-frame stop codon, then the to_stop argument
        has no effect:

        >>> coding_dna2 = Seq("TTGGCCATTGTAATGGGCCGC")
        >>> coding_dna2.translate()
        Seq('LAIVMGR')
        >>> coding_dna2.translate(to_stop=True)
        Seq('LAIVMGR')

        NOTE - Ambiguous codons like "TAN" or "NNN" could be an amino acid
        or a stop codon.  These are translated as "X".  Any invalid codon
        (e.g. "TA?" or "T-A") will throw a TranslationError.

        NOTE - This does NOT behave like the python string's translate
        method.  For that use str(my_seq).translate(...) instead
        """
        try:
            data = str(self)
        except UndefinedSequenceError:
            n = len(self)
            if n % 3 != 0:
                warnings.warn('Partial codon, len(sequence) not a multiple of three. This may become an error in future.', BiopythonWarning)
            return Seq(None, n // 3)
        return self.__class__(_translate_str(str(self), table, stop_symbol, to_stop, cds, gap=gap))

    def complement(self, inplace=False):
        """Return the complement as a DNA sequence.

        >>> Seq("CGA").complement()
        Seq('GCT')

        Any U in the sequence is treated as a T:

        >>> Seq("CGAUT").complement()
        Seq('GCTAA')

        In contrast, ``complement_rna`` returns an RNA sequence:

        >>> Seq("CGAUT").complement_rna()
        Seq('GCUAA')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.complement()
        MutableSeq('GCT')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.complement(inplace=True)
        MutableSeq('GCT')
        >>> my_seq
        MutableSeq('GCT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``complement_rna`` is called on a ``Seq`` object with ``inplace=True``.
        """
        ttable = _dna_complement_table
        try:
            data = self._data.translate(ttable)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def complement_rna(self, inplace=False):
        """Return the complement as an RNA sequence.

        >>> Seq("CGA").complement_rna()
        Seq('GCU')

        Any T in the sequence is treated as a U:

        >>> Seq("CGAUT").complement_rna()
        Seq('GCUAA')

        In contrast, ``complement`` returns a DNA sequence by default:

        >>> Seq("CGA").complement()
        Seq('GCT')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.complement_rna()
        MutableSeq('GCU')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.complement_rna(inplace=True)
        MutableSeq('GCU')
        >>> my_seq
        MutableSeq('GCU')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``complement_rna`` is called on a ``Seq`` object with ``inplace=True``.
        """
        try:
            data = self._data.translate(_rna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def reverse_complement(self, inplace=False):
        """Return the reverse complement as a DNA sequence.

        >>> Seq("CGA").reverse_complement()
        Seq('TCG')

        Any U in the sequence is treated as a T:

        >>> Seq("CGAUT").reverse_complement()
        Seq('AATCG')

        In contrast, ``reverse_complement_rna`` returns an RNA sequence:

        >>> Seq("CGA").reverse_complement_rna()
        Seq('UCG')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.reverse_complement()
        MutableSeq('TCG')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.reverse_complement(inplace=True)
        MutableSeq('TCG')
        >>> my_seq
        MutableSeq('TCG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``reverse_complement`` is called on a ``Seq`` object with
        ``inplace=True``.
        """
        try:
            data = self._data.translate(_dna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[::-1] = data
            return self
        return self.__class__(data[::-1])

    def reverse_complement_rna(self, inplace=False):
        """Return the reverse complement as an RNA sequence.

        >>> Seq("CGA").reverse_complement_rna()
        Seq('UCG')

        Any T in the sequence is treated as a U:

        >>> Seq("CGAUT").reverse_complement_rna()
        Seq('AAUCG')

        In contrast, ``reverse_complement`` returns a DNA sequence:

        >>> Seq("CGA").reverse_complement()
        Seq('TCG')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.reverse_complement_rna()
        MutableSeq('UCG')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.reverse_complement_rna(inplace=True)
        MutableSeq('UCG')
        >>> my_seq
        MutableSeq('UCG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``reverse_complement_rna`` is called on a ``Seq`` object with
        ``inplace=True``.
        """
        try:
            data = self._data.translate(_rna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[::-1] = data
            return self
        return self.__class__(data[::-1])

    def transcribe(self, inplace=False):
        """Transcribe a DNA sequence into RNA and return the RNA sequence as a new Seq object.

        Following the usual convention, the sequence is interpreted as the
        coding strand of the DNA double helix, not the template strand. This
        means we can get the RNA sequence just by switching T to U.

        >>> from Bio.Seq import Seq
        >>> coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
        >>> coding_dna
        Seq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> coding_dna.transcribe()
        Seq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')

        The sequence is modified in-place and returned if inplace is True:

        >>> sequence = MutableSeq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
        >>> sequence
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> sequence.transcribe()
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> sequence
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')

        >>> sequence.transcribe(inplace=True)
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> sequence
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``transcribe`` is called on a ``Seq`` object with ``inplace=True``.

        Trying to transcribe an RNA sequence has no effect.
        If you have a nucleotide sequence which might be DNA or RNA
        (or even a mixture), calling the transcribe method will ensure
        any T becomes U.

        Trying to transcribe a protein sequence will replace any
        T for Threonine with U for Selenocysteine, which has no
        biologically plausible rational.

        >>> from Bio.Seq import Seq
        >>> my_protein = Seq("MAIVMGRT")
        >>> my_protein.transcribe()
        Seq('MAIVMGRU')
        """
        data = self._data.replace(b'T', b'U').replace(b't', b'u')
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def back_transcribe(self, inplace=False):
        """Return the DNA sequence from an RNA sequence by creating a new Seq object.

        >>> from Bio.Seq import Seq
        >>> messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
        >>> messenger_rna
        Seq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> messenger_rna.back_transcribe()
        Seq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')

        The sequence is modified in-place and returned if inplace is True:

        >>> sequence = MutableSeq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
        >>> sequence
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> sequence.back_transcribe()
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> sequence
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')

        >>> sequence.back_transcribe(inplace=True)
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> sequence
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``transcribe`` is called on a ``Seq`` object with ``inplace=True``.

        Trying to back-transcribe DNA has no effect, If you have a nucleotide
        sequence which might be DNA or RNA (or even a mixture), calling the
        back-transcribe method will ensure any U becomes T.

        Trying to back-transcribe a protein sequence will replace any U for
        Selenocysteine with T for Threonine, which is biologically meaningless.

        >>> from Bio.Seq import Seq
        >>> my_protein = Seq("MAIVMGRU")
        >>> my_protein.back_transcribe()
        Seq('MAIVMGRT')
        """
        data = self._data.replace(b'U', b'T').replace(b'u', b't')
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def join(self, other):
        """Return a merge of the sequences in other, spaced by the sequence from self.

        Accepts a Seq object, MutableSeq object, or string (and iterates over
        the letters), or an iterable containing Seq, MutableSeq, or string
        objects. These arguments will be concatenated with the calling sequence
        as the spacer:

        >>> concatenated = Seq('NNNNN').join([Seq("AAA"), Seq("TTT"), Seq("PPP")])
        >>> concatenated
        Seq('AAANNNNNTTTNNNNNPPP')

        Joining the letters of a single sequence:

        >>> Seq('NNNNN').join(Seq("ACGT"))
        Seq('ANNNNNCNNNNNGNNNNNT')
        >>> Seq('NNNNN').join("ACGT")
        Seq('ANNNNNCNNNNNGNNNNNT')
        """
        if isinstance(other, _SeqAbstractBaseClass):
            return self.__class__(str(self).join(str(other)))
        elif isinstance(other, str):
            return self.__class__(str(self).join(other))
        from Bio.SeqRecord import SeqRecord
        if isinstance(other, SeqRecord):
            raise TypeError('Iterable cannot be a SeqRecord')
        for c in other:
            if isinstance(c, SeqRecord):
                raise TypeError('Iterable cannot contain SeqRecords')
            elif not isinstance(c, (str, _SeqAbstractBaseClass)):
                raise TypeError('Input must be an iterable of Seq objects, MutableSeq objects, or strings')
        return self.__class__(str(self).join([str(_) for _ in other]))

    def replace(self, old, new, inplace=False):
        """Return a copy with all occurrences of subsequence old replaced by new.

        >>> s = Seq("ACGTAACCGGTT")
        >>> t = s.replace("AC", "XYZ")
        >>> s
        Seq('ACGTAACCGGTT')
        >>> t
        Seq('XYZGTAXYZCGGTT')

        For mutable sequences, passing inplace=True will modify the sequence in place:

        >>> m = MutableSeq("ACGTAACCGGTT")
        >>> t = m.replace("AC", "XYZ")
        >>> m
        MutableSeq('ACGTAACCGGTT')
        >>> t
        MutableSeq('XYZGTAXYZCGGTT')

        >>> m = MutableSeq("ACGTAACCGGTT")
        >>> t = m.replace("AC", "XYZ", inplace=True)
        >>> m
        MutableSeq('XYZGTAXYZCGGTT')
        >>> t
        MutableSeq('XYZGTAXYZCGGTT')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``replace`` is called on a ``Seq`` object with ``inplace=True``.
        """
        if isinstance(old, _SeqAbstractBaseClass):
            old = bytes(old)
        elif isinstance(old, str):
            old = old.encode('ASCII')
        if isinstance(new, _SeqAbstractBaseClass):
            new = bytes(new)
        elif isinstance(new, str):
            new = new.encode('ASCII')
        data = self._data.replace(old, new)
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    @property
    def defined(self):
        """Return True if the sequence is defined, False if undefined or partially defined.

        Zero-length sequences are always considered to be defined.
        """
        if isinstance(self._data, (bytes, bytearray)):
            return True
        else:
            return self._data.defined

    @property
    def defined_ranges(self):
        """Return a tuple of the ranges where the sequence contents is defined.

        The return value has the format ((start1, end1), (start2, end2), ...).
        """
        if isinstance(self._data, (bytes, bytearray)):
            length = len(self)
            if length > 0:
                return ((0, length),)
            else:
                return ()
        else:
            return self._data.defined_ranges