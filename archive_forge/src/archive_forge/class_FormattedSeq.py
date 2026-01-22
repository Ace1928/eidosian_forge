import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
class FormattedSeq:
    """A linear or circular sequence object for restriction analysis.

    Translates a Bio.Seq into a formatted sequence to be used with Restriction.

    Roughly: remove anything which is not IUPAC alphabet and then add a space
             in front of the sequence to get a biological index instead of a
             python index (i.e. index of the first base is 1 not 0).

    Retains information about the shape of the molecule linear (default) or
    circular. Restriction sites are search over the edges of circular sequence.
    """
    _remove_chars = string.whitespace.encode() + string.digits.encode()
    _table = _make_FormattedSeq_table()

    def __init__(self, seq, linear=True):
        """Initialize ``FormattedSeq`` with sequence and topology (optional).

        ``seq`` is either a ``Bio.Seq``, ``Bio.MutableSeq`` or a
        ``FormattedSeq``. If ``seq`` is a ``FormattedSeq``, ``linear``
        will have no effect on the shape of the sequence.
        """
        if isinstance(seq, (Seq, MutableSeq)):
            self.lower = seq.islower()
            data = bytes(seq)
            self.data = data.translate(self._table, delete=self._remove_chars)
            if 0 in self.data:
                raise TypeError(f'Invalid character found in {data.decode()}')
            self.data = ' ' + self.data.decode('ASCII')
            self.linear = linear
            self.klass = seq.__class__
        elif isinstance(seq, FormattedSeq):
            self.lower = seq.lower
            self.data = seq.data
            self.linear = seq.linear
            self.klass = seq.klass
        else:
            raise TypeError(f'expected Seq or MutableSeq, got {type(seq)}')

    def __len__(self):
        """Return length of ``FormattedSeq``.

        ``FormattedSeq`` has a leading space, thus subtract 1.
        """
        return len(self.data) - 1

    def __repr__(self):
        """Represent ``FormattedSeq`` class as a string."""
        return f'FormattedSeq({self[1:]!r}, linear={self.linear!r})'

    def __eq__(self, other):
        """Implement equality operator for ``FormattedSeq`` object."""
        if isinstance(other, FormattedSeq):
            if repr(self) == repr(other):
                return True
            else:
                return False
        return False

    def circularise(self):
        """Circularise sequence in place."""
        self.linear = False

    def linearise(self):
        """Linearise sequence in place."""
        self.linear = True

    def to_linear(self):
        """Make a new instance of sequence as linear."""
        new = self.__class__(self)
        new.linear = True
        return new

    def to_circular(self):
        """Make a new instance of sequence as circular."""
        new = self.__class__(self)
        new.linear = False
        return new

    def is_linear(self):
        """Return if sequence is linear (True) or circular (False)."""
        return self.linear

    def finditer(self, pattern, size):
        """Return a list of a given pattern which occurs in the sequence.

        The list is made of tuple (location, pattern.group).
        The latter is used with non palindromic sites.
        Pattern is the regular expression pattern corresponding to the
        enzyme restriction site.
        Size is the size of the restriction enzyme recognition-site size.
        """
        if self.is_linear():
            data = self.data
        else:
            data = self.data + self.data[1:size]
        return [(i.start(), i.group) for i in re.finditer(pattern, data)]

    def __getitem__(self, i):
        """Return substring of ``FormattedSeq``.

        The class of the returned object is the class of the respective
        sequence. Note that due to the leading space, indexing is 1-based:

        >>> from Bio.Seq import Seq
        >>> from Bio.Restriction.Restriction import FormattedSeq
        >>> f_seq = FormattedSeq(Seq('ATGCATGC'))
        >>> f_seq[1]
        Seq('A')

        """
        if self.lower:
            return self.klass(self.data[i].lower())
        return self.klass(self.data[i])