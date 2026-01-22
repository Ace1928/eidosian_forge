import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class SeqFeature:
    """Represent a Sequence Feature on an object.

    Attributes:
     - location - the location of the feature on the sequence (SimpleLocation)
     - type - the specified type of the feature (ie. CDS, exon, repeat...)
     - id - A string identifier for the feature.
     - qualifiers - A dictionary of qualifiers on the feature. These are
       analogous to the qualifiers from a GenBank feature table. The keys of
       the dictionary are qualifier names, the values are the qualifier
       values.

    """

    def __init__(self, location=None, type='', id='<unknown id>', qualifiers=None, sub_features=None):
        """Initialize a SeqFeature on a sequence.

        location can either be a SimpleLocation (with strand argument also
        given if required), or None.

        e.g. With no strand, on the forward strand, and on the reverse strand:

        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> f1 = SeqFeature(SimpleLocation(5, 10), type="domain")
        >>> f1.location.strand == None
        True
        >>> f2 = SeqFeature(SimpleLocation(7, 110, strand=1), type="CDS")
        >>> f2.location.strand == +1
        True
        >>> f3 = SeqFeature(SimpleLocation(9, 108, strand=-1), type="CDS")
        >>> f3.location.strand == -1
        True

        For exact start/end positions, an integer can be used (as shown above)
        as shorthand for the ExactPosition object. For non-exact locations, the
        SimpleLocation must be specified via the appropriate position objects.
        """
        if location is not None and (not isinstance(location, SimpleLocation)) and (not isinstance(location, CompoundLocation)):
            raise TypeError('SimpleLocation, CompoundLocation (or None) required for the location')
        self.location = location
        self.type = type
        self.id = id
        self.qualifiers = {}
        if qualifiers is not None:
            self.qualifiers.update(qualifiers)
        if sub_features is not None:
            raise TypeError('Rather than sub_features, use a CompoundLocation')

    def _get_strand(self):
        """Get function for the strand property (PRIVATE)."""
        warnings.warn('Please use .location.strand rather than .strand', BiopythonDeprecationWarning)
        return self.location.strand

    def _set_strand(self, value):
        """Set function for the strand property (PRIVATE)."""
        warnings.warn('Please use .location.strand rather than .strand', BiopythonDeprecationWarning)
        try:
            self.location.strand = value
        except AttributeError:
            if self.location is None:
                if value is not None:
                    raise ValueError("Can't set strand without a location.") from None
            else:
                raise
    strand = property(fget=_get_strand, fset=_set_strand, doc="Alias for the location's strand (DEPRECATED).")

    def _get_ref(self):
        """Get function for the reference property (PRIVATE)."""
        warnings.warn('Please use .location.ref rather than .ref', BiopythonDeprecationWarning)
        try:
            return self.location.ref
        except AttributeError:
            return None

    def _set_ref(self, value):
        """Set function for the reference property (PRIVATE)."""
        warnings.warn('Please use .location.ref rather than .ref', BiopythonDeprecationWarning)
        try:
            self.location.ref = value
        except AttributeError:
            if self.location is None:
                if value is not None:
                    raise ValueError("Can't set ref without a location.") from None
            else:
                raise
    ref = property(fget=_get_ref, fset=_set_ref, doc="Alias for the location's ref (DEPRECATED).")

    def _get_ref_db(self):
        """Get function for the database reference property (PRIVATE)."""
        warnings.warn('Please use .location.ref_db rather than .ref_db', BiopythonDeprecationWarning)
        try:
            return self.location.ref_db
        except AttributeError:
            return None

    def _set_ref_db(self, value):
        """Set function for the database reference property (PRIVATE)."""
        warnings.warn('Please use .location.ref_db rather than .ref_db', BiopythonDeprecationWarning)
        self.location.ref_db = value
    ref_db = property(fget=_get_ref_db, fset=_set_ref_db, doc="Alias for the location's ref_db (DEPRECATED).")

    def __eq__(self, other):
        """Check if two SeqFeature objects should be considered equal."""
        return isinstance(other, SeqFeature) and self.id == other.id and (self.type == other.type) and (self.location == other.location) and (self.qualifiers == other.qualifiers)

    def __repr__(self):
        """Represent the feature as a string for debugging."""
        answer = f'{self.__class__.__name__}({self.location!r}'
        if self.type:
            answer += f', type={self.type!r}'
        if self.id and self.id != '<unknown id>':
            answer += f', id={self.id!r}'
        if self.qualifiers:
            answer += ', qualifiers=...'
        answer += ')'
        return answer

    def __str__(self):
        """Return the full feature as a python string."""
        out = f'type: {self.type}\n'
        out += f'location: {self.location}\n'
        if self.id and self.id != '<unknown id>':
            out += f'id: {self.id}\n'
        out += 'qualifiers:\n'
        for qual_key in sorted(self.qualifiers):
            out += f'    Key: {qual_key}, Value: {self.qualifiers[qual_key]}\n'
        return out

    def _shift(self, offset):
        """Return a copy of the feature with its location shifted (PRIVATE).

        The annotation qualifiers are copied.
        """
        return SeqFeature(location=self.location._shift(offset), type=self.type, id=self.id, qualifiers=self.qualifiers.copy())

    def _flip(self, length):
        """Return a copy of the feature with its location flipped (PRIVATE).

        The argument length gives the length of the parent sequence. For
        example a location 0..20 (+1 strand) with parent length 30 becomes
        after flipping 10..30 (-1 strand). Strandless (None) or unknown
        strand (0) remain like that - just their end points are changed.

        The annotation qualifiers are copied.
        """
        return SeqFeature(location=self.location._flip(length), type=self.type, id=self.id, qualifiers=self.qualifiers.copy())

    def extract(self, parent_sequence, references=None):
        """Extract the feature's sequence from supplied parent sequence.

        The parent_sequence can be a Seq like object or a string, and will
        generally return an object of the same type. The exception to this is
        a MutableSeq as the parent sequence will return a Seq object.

        This should cope with complex locations including complements, joins
        and fuzzy positions. Even mixed strand features should work! This
        also covers features on protein sequences (e.g. domains), although
        here reverse strand features are not permitted. If the
        location refers to other records, they must be supplied in the
        optional dictionary references.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")
        >>> f = SeqFeature(SimpleLocation(8, 15), type="domain")
        >>> f.extract(seq)
        Seq('VALIVIC')

        If the SimpleLocation is None, e.g. when parsing invalid locus
        locations in the GenBank parser, extract() will raise a ValueError.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SeqFeature
        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")
        >>> f = SeqFeature(None, type="domain")
        >>> f.extract(seq)
        Traceback (most recent call last):
           ...
        ValueError: The feature's .location is None. Check the sequence file for a valid location.

        Note - currently only compound features of type "join" are supported.
        """
        if self.location is None:
            raise ValueError("The feature's .location is None. Check the sequence file for a valid location.")
        return self.location.extract(parent_sequence, references=references)

    def translate(self, parent_sequence, table='Standard', start_offset=None, stop_symbol='*', to_stop=False, cds=None, gap=None):
        """Get a translation of the feature's sequence.

        This method is intended for CDS or other features that code proteins
        and is a shortcut that will both extract the feature and
        translate it, taking into account the codon_start and transl_table
        qualifiers, if they are present. If they are not present the
        value of the arguments "table" and "start_offset" are used.

        The "cds" parameter is set to "True" if the feature is of type
        "CDS" but can be overridden by giving an explicit argument.

        The arguments stop_symbol, to_stop and gap have the same meaning
        as Seq.translate, refer to that documentation for further information.

        Arguments:
         - parent_sequence - A DNA or RNA sequence.
         - table - Which codon table to use if there is no transl_table
           qualifier for this feature. This can be either a name
           (string), an NCBI identifier (integer), or a CodonTable
           object (useful for non-standard genetic codes).  This
           defaults to the "Standard" table.
         - start_offset - offset at which the first complete codon of a
           coding feature can be found, relative to the first base of
           that feature. Has a valid value of 0, 1 or 2. NOTE: this
           uses python's 0-based numbering whereas the codon_start
           qualifier in files from NCBI use 1-based numbering.
           Will override a codon_start qualifier

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> seq = Seq("GGTTACACTTACCGATAATGTCTCTGATGA")
        >>> f = SeqFeature(SimpleLocation(0, 30), type="CDS")
        >>> f.qualifiers['transl_table'] = [11]

        Note that features of type CDS are subject to the usual
        checks at translation. But you can override this behavior
        by giving explicit arguments:

        >>> f.translate(seq, cds=False)
        Seq('GYTYR*CL**')

        Now use the start_offset argument to change the frame. Note
        this uses python 0-based numbering.

        >>> f.translate(seq, start_offset=1, cds=False)
        Seq('VTLTDNVSD')

        Alternatively use the codon_start qualifier to do the same
        thing. Note: this uses 1-based numbering, which is found
        in files from NCBI.

        >>> f.qualifiers['codon_start'] = [2]
        >>> f.translate(seq, cds=False)
        Seq('VTLTDNVSD')
        """
        if start_offset is None:
            try:
                start_offset = int(self.qualifiers['codon_start'][0]) - 1
            except KeyError:
                start_offset = 0
        if start_offset not in [0, 1, 2]:
            raise ValueError(f'The start_offset must be 0, 1, or 2. The supplied value is {start_offset}. Check the value of either the codon_start qualifier or the start_offset argument')
        feat_seq = self.extract(parent_sequence)[start_offset:]
        codon_table = self.qualifiers.get('transl_table', [table])[0]
        if cds is None:
            cds = self.type == 'CDS'
        return feat_seq.translate(table=codon_table, stop_symbol=stop_symbol, to_stop=to_stop, cds=cds, gap=gap)

    def __bool__(self):
        """Boolean value of an instance of this class (True).

        This behavior is for backwards compatibility, since until the
        __len__ method was added, a SeqFeature always evaluated as True.

        Note that in comparison, Seq objects, strings, lists, etc, will all
        evaluate to False if they have length zero.

        WARNING: The SeqFeature may in future evaluate to False when its
        length is zero (in order to better match normal python behavior)!
        """
        return True

    def __len__(self):
        """Return the length of the region where the feature is located.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")
        >>> f = SeqFeature(SimpleLocation(8, 15), type="domain")
        >>> len(f)
        7
        >>> f.extract(seq)
        Seq('VALIVIC')
        >>> len(f.extract(seq))
        7

        This is a proxy for taking the length of the feature's location:

        >>> len(f.location)
        7

        For simple features this is the same as the region spanned (end
        position minus start position using Pythonic counting). However, for
        a compound location (e.g. a CDS as the join of several exons) the
        gaps are not counted (e.g. introns). This ensures that len(f) matches
        len(f.extract(parent_seq)), and also makes sure things work properly
        with features wrapping the origin etc.
        """
        return len(self.location)

    def __iter__(self):
        """Iterate over the parent positions within the feature.

        The iteration order is strand aware, and can be thought of as moving
        along the feature using the parent sequence coordinates:

        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> f = SeqFeature(SimpleLocation(5, 10, strand=-1), type="domain")
        >>> len(f)
        5
        >>> for i in f: print(i)
        9
        8
        7
        6
        5
        >>> list(f)
        [9, 8, 7, 6, 5]

        This is a proxy for iterating over the location,

        >>> list(f.location)
        [9, 8, 7, 6, 5]
        """
        return iter(self.location)

    def __contains__(self, value):
        """Check if an integer position is within the feature.

        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> f = SeqFeature(SimpleLocation(5, 10, strand=-1), type="domain")
        >>> len(f)
        5
        >>> [i for i in range(15) if i in f]
        [5, 6, 7, 8, 9]

        For example, to see which features include a SNP position, you could
        use this:

        >>> from Bio import SeqIO
        >>> record = SeqIO.read("GenBank/NC_000932.gb", "gb")
        >>> for f in record.features:
        ...     if 1750 in f:
        ...         print("%s %s" % (f.type, f.location))
        source [0:154478](+)
        gene [1716:4347](-)
        tRNA join{[4310:4347](-), [1716:1751](-)}

        Note that for a feature defined as a join of several subfeatures (e.g.
        the union of several exons) the gaps are not checked (e.g. introns).
        In this example, the tRNA location is defined in the GenBank file as
        complement(join(1717..1751,4311..4347)), so that position 1760 falls
        in the gap:

        >>> for f in record.features:
        ...     if 1760 in f:
        ...         print("%s %s" % (f.type, f.location))
        source [0:154478](+)
        gene [1716:4347](-)

        Note that additional care may be required with fuzzy locations, for
        example just before a BeforePosition:

        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation
        >>> from Bio.SeqFeature import BeforePosition
        >>> f = SeqFeature(SimpleLocation(BeforePosition(3), 8), type="domain")
        >>> len(f)
        5
        >>> [i for i in range(10) if i in f]
        [3, 4, 5, 6, 7]

        Note that is is a proxy for testing membership on the location.

        >>> [i for i in range(10) if i in f.location]
        [3, 4, 5, 6, 7]
        """
        return value in self.location