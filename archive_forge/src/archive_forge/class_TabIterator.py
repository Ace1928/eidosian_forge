from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class TabIterator(SequenceIterator):
    """Parser for tab-delimited files."""

    def __init__(self, source):
        """Iterate over tab separated lines as SeqRecord objects.

        Each line of the file should contain one tab only, dividing the line
        into an identifier and the full sequence.

        Arguments:
         - source - file-like object opened in text mode, or a path to a file

        The first field is taken as the record's .id and .name (regardless of
        any spaces within the text) and the second field is the sequence.

        Any blank lines are ignored.

        Examples
        --------
        >>> with open("GenBank/NC_005816.tsv") as handle:
        ...     for record in TabIterator(handle):
        ...         print("%s length %i" % (record.id, len(record)))
        gi|45478712|ref|NP_995567.1| length 340
        gi|45478713|ref|NP_995568.1| length 260
        gi|45478714|ref|NP_995569.1| length 64
        gi|45478715|ref|NP_995570.1| length 123
        gi|45478716|ref|NP_995571.1| length 145
        gi|45478717|ref|NP_995572.1| length 357
        gi|45478718|ref|NP_995573.1| length 138
        gi|45478719|ref|NP_995574.1| length 312
        gi|45478720|ref|NP_995575.1| length 99
        gi|45478721|ref|NP_995576.1| length 90

        """
        super().__init__(source, mode='t', fmt='Tab-separated plain-text')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        for line in handle:
            try:
                title, seq = line.split('\t')
            except ValueError:
                if line.strip() == '':
                    continue
                raise ValueError('Each line should have one tab separating the' + ' title and sequence, this line has %i tabs: %r' % (line.count('\t'), line)) from None
            title = title.strip()
            seq = seq.strip()
            yield SeqRecord(Seq(seq), id=title, name=title, description='')