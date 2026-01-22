import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class SimpleLocation(Location):
    """Specify the location of a feature along a sequence.

    The SimpleLocation is used for simple continuous features, which can
    be described as running from a start position to and end position
    (optionally with a strand and reference information).  More complex
    locations made up from several non-continuous parts (e.g. a coding
    sequence made up of several exons) are described using a SeqFeature
    with a CompoundLocation.

    Note that the start and end location numbering follow Python's scheme,
    thus a GenBank entry of 123..150 (one based counting) becomes a location
    of [122:150] (zero based counting).

    >>> from Bio.SeqFeature import SimpleLocation
    >>> f = SimpleLocation(122, 150)
    >>> print(f)
    [122:150]
    >>> print(f.start)
    122
    >>> print(f.end)
    150
    >>> print(f.strand)
    None

    Note the strand defaults to None. If you are working with nucleotide
    sequences you'd want to be explicit if it is the forward strand:

    >>> from Bio.SeqFeature import SimpleLocation
    >>> f = SimpleLocation(122, 150, strand=+1)
    >>> print(f)
    [122:150](+)
    >>> print(f.strand)
    1

    Note that for a parent sequence of length n, the SimpleLocation
    start and end must satisfy the inequality 0 <= start <= end <= n.
    This means even for features on the reverse strand of a nucleotide
    sequence, we expect the 'start' coordinate to be less than the
    'end'.

    >>> from Bio.SeqFeature import SimpleLocation
    >>> r = SimpleLocation(122, 150, strand=-1)
    >>> print(r)
    [122:150](-)
    >>> print(r.start)
    122
    >>> print(r.end)
    150
    >>> print(r.strand)
    -1

    i.e. Rather than thinking of the 'start' and 'end' biologically in a
    strand aware manner, think of them as the 'left most' or 'minimum'
    boundary, and the 'right most' or 'maximum' boundary of the region
    being described. This is particularly important with compound
    locations describing non-continuous regions.

    In the example above we have used standard exact positions, but there
    are also specialised position objects used to represent fuzzy positions
    as well, for example a GenBank location like complement(<123..150)
    would use a BeforePosition object for the start.
    """

    def __init__(self, start, end, strand=None, ref=None, ref_db=None):
        """Initialize the class.

        start and end arguments specify the values where the feature begins
        and ends. These can either by any of the ``*Position`` objects that
        inherit from Position, or can just be integers specifying the position.
        In the case of integers, the values are assumed to be exact and are
        converted in ExactPosition arguments. This is meant to make it easy
        to deal with non-fuzzy ends.

        i.e. Short form:

        >>> from Bio.SeqFeature import SimpleLocation
        >>> loc = SimpleLocation(5, 10, strand=-1)
        >>> print(loc)
        [5:10](-)

        Explicit form:

        >>> from Bio.SeqFeature import SimpleLocation, ExactPosition
        >>> loc = SimpleLocation(ExactPosition(5), ExactPosition(10), strand=-1)
        >>> print(loc)
        [5:10](-)

        Other fuzzy positions are used similarly,

        >>> from Bio.SeqFeature import SimpleLocation
        >>> from Bio.SeqFeature import BeforePosition, AfterPosition
        >>> loc2 = SimpleLocation(BeforePosition(5), AfterPosition(10), strand=-1)
        >>> print(loc2)
        [<5:>10](-)

        For nucleotide features you will also want to specify the strand,
        use 1 for the forward (plus) strand, -1 for the reverse (negative)
        strand, 0 for stranded but strand unknown (? in GFF3), or None for
        when the strand does not apply (dot in GFF3), e.g. features on
        proteins.

        >>> loc = SimpleLocation(5, 10, strand=+1)
        >>> print(loc)
        [5:10](+)
        >>> print(loc.strand)
        1

        Normally feature locations are given relative to the parent
        sequence you are working with, but an explicit accession can
        be given with the optional ref and db_ref strings:

        >>> loc = SimpleLocation(105172, 108462, ref="AL391218.9", strand=1)
        >>> print(loc)
        AL391218.9[105172:108462](+)
        >>> print(loc.ref)
        AL391218.9

        """
        if isinstance(start, Position):
            self._start = start
        elif isinstance(start, int):
            self._start = ExactPosition(start)
        else:
            raise TypeError(f'start={start!r} {type(start)}')
        if isinstance(end, Position):
            self._end = end
        elif isinstance(end, int):
            self._end = ExactPosition(end)
        else:
            raise TypeError(f'end={end!r} {type(end)}')
        if isinstance(self.start, int) and isinstance(self.end, int) and (self.start > self.end):
            raise ValueError(f'End location ({self.end}) must be greater than or equal to start location ({self.start})')
        self.strand = strand
        self.ref = ref
        self.ref_db = ref_db

    @staticmethod
    def fromstring(text, length=None, circular=False):
        """Create a SimpleLocation object from a string."""
        if text.startswith('complement('):
            text = text[11:-1]
            strand = -1
        else:
            strand = None
        try:
            s, e = text.split('..')
            s = int(s) - 1
            e = int(e)
        except ValueError:
            pass
        else:
            if 0 <= s < e:
                return SimpleLocation(s, e, strand)
        try:
            ref, text = text.split(':')
        except ValueError:
            ref = None
        m = _re_location_category.match(text)
        if m is None:
            raise LocationParserError(f"Could not parse feature location '{text}'")
        for key, value in m.groupdict().items():
            if value is not None:
                break
        assert value == text
        if key == 'bond':
            warnings.warn('Dropping bond qualifier in feature location', BiopythonParserWarning)
            text = text[5:-1]
            s_pos = Position.fromstring(text, -1)
            e_pos = Position.fromstring(text)
        elif key == 'solo':
            s_pos = Position.fromstring(text, -1)
            e_pos = Position.fromstring(text)
        elif key in ('pair', 'within', 'oneof'):
            s, e = text.split('..')
            s_pos = Position.fromstring(s, -1)
            e_pos = Position.fromstring(e)
            if s_pos >= e_pos:
                if not circular:
                    raise LocationParserError(f"it appears that '{text}' is a feature that spans the origin, but the sequence topology is undefined")
                warnings.warn('Attempting to fix invalid location %r as it looks like incorrect origin wrapping. Please fix input file, this could have unintended behavior.' % text, BiopythonParserWarning)
                f1 = SimpleLocation(s_pos, length, strand)
                f2 = SimpleLocation(0, e_pos, strand)
                if strand == -1:
                    return f2 + f1
                else:
                    return f1 + f2
        elif key == 'between':
            s, e = text.split('^')
            s = int(s)
            e = int(e)
            if s + 1 == e or (s == length and e == 1):
                s_pos = ExactPosition(s)
                e_pos = s_pos
            else:
                raise LocationParserError(f"invalid feature location '{text}'")
        if s_pos < 0:
            raise LocationParserError(f"negative starting position in feature location '{text}'")
        return SimpleLocation(s_pos, e_pos, strand, ref=ref)

    def _get_strand(self):
        """Get function for the strand property (PRIVATE)."""
        return self._strand

    def _set_strand(self, value):
        """Set function for the strand property (PRIVATE)."""
        if value not in [+1, -1, 0, None]:
            raise ValueError(f'Strand should be +1, -1, 0 or None, not {value!r}')
        self._strand = value
    strand = property(fget=_get_strand, fset=_set_strand, doc='Strand of the location (+1, -1, 0 or None).')

    def __str__(self):
        """Return a representation of the SimpleLocation object (with python counting).

        For the simple case this uses the python splicing syntax, [122:150]
        (zero based counting) which GenBank would call 123..150 (one based
        counting).
        """
        answer = f'[{self._start}:{self._end}]'
        if self.ref and self.ref_db:
            answer = f'{self.ref_db}:{self.ref}{answer}'
        elif self.ref:
            answer = self.ref + answer
        if self.strand is None:
            return answer
        elif self.strand == +1:
            return answer + '(+)'
        elif self.strand == -1:
            return answer + '(-)'
        else:
            return answer + '(?)'

    def __repr__(self):
        """Represent the SimpleLocation object as a string for debugging."""
        optional = ''
        if self.strand is not None:
            optional += f', strand={self.strand!r}'
        if self.ref is not None:
            optional += f', ref={self.ref!r}'
        if self.ref_db is not None:
            optional += f', ref_db={self.ref_db!r}'
        return f'{self.__class__.__name__}({self.start!r}, {self.end!r}{optional})'

    def __add__(self, other):
        """Combine location with another SimpleLocation object, or shift it.

        You can add two feature locations to make a join CompoundLocation:

        >>> from Bio.SeqFeature import SimpleLocation
        >>> f1 = SimpleLocation(5, 10)
        >>> f2 = SimpleLocation(20, 30)
        >>> combined = f1 + f2
        >>> print(combined)
        join{[5:10], [20:30]}

        This is thus equivalent to:

        >>> from Bio.SeqFeature import CompoundLocation
        >>> join = CompoundLocation([f1, f2])
        >>> print(join)
        join{[5:10], [20:30]}

        You can also use sum(...) in this way:

        >>> join = sum([f1, f2])
        >>> print(join)
        join{[5:10], [20:30]}

        Furthermore, you can combine a SimpleLocation with a CompoundLocation
        in this way.

        Separately, adding an integer will give a new SimpleLocation with
        its start and end offset by that amount. For example:

        >>> print(f1)
        [5:10]
        >>> print(f1 + 100)
        [105:110]
        >>> print(200 + f1)
        [205:210]

        This can be useful when editing annotation.
        """
        if isinstance(other, SimpleLocation):
            return CompoundLocation([self, other])
        elif isinstance(other, int):
            return self._shift(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Return a SimpleLocation object by shifting the location by an integer amount."""
        if isinstance(other, int):
            return self._shift(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtracting an integer will shift the start and end by that amount.

        >>> from Bio.SeqFeature import SimpleLocation
        >>> f1 = SimpleLocation(105, 150)
        >>> print(f1)
        [105:150]
        >>> print(f1 - 100)
        [5:50]

        This can be useful when editing annotation. You can also add an integer
        to a feature location (which shifts in the opposite direction).
        """
        if isinstance(other, int):
            return self._shift(-other)
        else:
            return NotImplemented

    def __nonzero__(self):
        """Return True regardless of the length of the feature.

        This behavior is for backwards compatibility, since until the
        __len__ method was added, a SimpleLocation always evaluated as True.

        Note that in comparison, Seq objects, strings, lists, etc, will all
        evaluate to False if they have length zero.

        WARNING: The SimpleLocation may in future evaluate to False when its
        length is zero (in order to better match normal python behavior)!
        """
        return True

    def __len__(self):
        """Return the length of the region described by the SimpleLocation object.

        Note that extra care may be needed for fuzzy locations, e.g.

        >>> from Bio.SeqFeature import SimpleLocation
        >>> from Bio.SeqFeature import BeforePosition, AfterPosition
        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))
        >>> len(loc)
        5
        """
        return int(self._end) - int(self._start)

    def __contains__(self, value):
        """Check if an integer position is within the SimpleLocation object.

        Note that extra care may be needed for fuzzy locations, e.g.

        >>> from Bio.SeqFeature import SimpleLocation
        >>> from Bio.SeqFeature import BeforePosition, AfterPosition
        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))
        >>> len(loc)
        5
        >>> [i for i in range(15) if i in loc]
        [5, 6, 7, 8, 9]
        """
        if not isinstance(value, int):
            raise ValueError('Currently we only support checking for integer positions being within a SimpleLocation.')
        if value < self._start or value >= self._end:
            return False
        else:
            return True

    def __iter__(self):
        """Iterate over the parent positions within the SimpleLocation object.

        >>> from Bio.SeqFeature import SimpleLocation
        >>> from Bio.SeqFeature import BeforePosition, AfterPosition
        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))
        >>> len(loc)
        5
        >>> for i in loc: print(i)
        5
        6
        7
        8
        9
        >>> list(loc)
        [5, 6, 7, 8, 9]
        >>> [i for i in range(15) if i in loc]
        [5, 6, 7, 8, 9]

        Note this is strand aware:

        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10), strand = -1)
        >>> list(loc)
        [9, 8, 7, 6, 5]
        """
        if self.strand == -1:
            yield from range(self._end - 1, self._start - 1, -1)
        else:
            yield from range(self._start, self._end)

    def __eq__(self, other):
        """Implement equality by comparing all the location attributes."""
        if not isinstance(other, SimpleLocation):
            return False
        return self._start == other.start and self._end == other.end and (self._strand == other.strand) and (self.ref == other.ref) and (self.ref_db == other.ref_db)

    def _shift(self, offset):
        """Return a copy of the SimpleLocation shifted by an offset (PRIVATE).

        Returns self when location is relative to an external reference.
        """
        if self.ref or self.ref_db:
            return self
        return SimpleLocation(start=self._start + offset, end=self._end + offset, strand=self.strand)

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE).

        Returns self when location is relative to an external reference.
        """
        if self.ref or self.ref_db:
            return self
        if self.strand == +1:
            flip_strand = -1
        elif self.strand == -1:
            flip_strand = +1
        else:
            flip_strand = self.strand
        return SimpleLocation(start=self._end._flip(length), end=self._start._flip(length), strand=flip_strand)

    @property
    def parts(self):
        """Read only list of sections (always one, the SimpleLocation object).

        This is a convenience property allowing you to write code handling
        both SimpleLocation objects (with one part) and more complex
        CompoundLocation objects (with multiple parts) interchangeably.
        """
        return [self]

    @property
    def start(self):
        """Start location - left most (minimum) value, regardless of strand.

        Read only, returns an integer like position object, possibly a fuzzy
        position.
        """
        return self._start

    @property
    def end(self):
        """End location - right most (maximum) value, regardless of strand.

        Read only, returns an integer like position object, possibly a fuzzy
        position.
        """
        return self._end

    def extract(self, parent_sequence, references=None):
        """Extract the sequence from supplied parent sequence using the SimpleLocation object.

        The parent_sequence can be a Seq like object or a string, and will
        generally return an object of the same type. The exception to this is
        a MutableSeq as the parent sequence will return a Seq object.
        If the location refers to other records, they must be supplied
        in the optional dictionary references.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SimpleLocation
        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")
        >>> feature_loc = SimpleLocation(8, 15)
        >>> feature_loc.extract(seq)
        Seq('VALIVIC')

        """
        if self.ref or self.ref_db:
            if not references:
                raise ValueError(f'Feature references another sequence ({self.ref}), references mandatory')
            elif self.ref not in references:
                raise ValueError(f'Feature references another sequence ({self.ref}), not found in references')
            parent_sequence = references[self.ref]
        f_seq = parent_sequence[int(self.start):int(self.end)]
        if isinstance(f_seq, MutableSeq):
            f_seq = Seq(f_seq)
        if self.strand == -1:
            f_seq = reverse_complement(f_seq)
        return f_seq