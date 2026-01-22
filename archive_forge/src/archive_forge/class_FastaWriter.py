import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
class FastaWriter(SequenceWriter):
    """Class to write Fasta format files (OBSOLETE).

    Please use the ``as_fasta`` function instead, or the top level
    ``Bio.SeqIO.write()`` function instead using ``format="fasta"``.
    """

    def __init__(self, target, wrap=60, record2title=None):
        """Create a Fasta writer (OBSOLETE).

        Arguments:
         - target - Output stream opened in text mode, or a path to a file.
         - wrap -   Optional line length used to wrap sequence lines.
           Defaults to wrapping the sequence at 60 characters
           Use zero (or None) for no wrapping, giving a single
           long line for the sequence.
         - record2title - Optional function to return the text to be
           used for the title line of each record.  By default
           a combination of the record.id and record.description
           is used.  If the record.description starts with the
           record.id, then just the record.description is used.

        You can either use::

            handle = open(filename, "w")
            writer = FastaWriter(handle)
            writer.write_file(myRecords)
            handle.close()

        Or, follow the sequential file writer system, for example::

            handle = open(filename, "w")
            writer = FastaWriter(handle)
            writer.write_header() # does nothing for Fasta files
            ...
            Multiple writer.write_record() and/or writer.write_records() calls
            ...
            writer.write_footer() # does nothing for Fasta files
            handle.close()

        """
        super().__init__(target)
        if wrap:
            if wrap < 1:
                raise ValueError
        self.wrap = wrap
        self.record2title = record2title

    def write_record(self, record):
        """Write a single Fasta record to the file."""
        if self.record2title:
            title = self.clean(self.record2title(record))
        else:
            id = self.clean(record.id)
            description = self.clean(record.description)
            if description and description.split(None, 1)[0] == id:
                title = description
            elif description:
                title = f'{id} {description}'
            else:
                title = id
        assert '\n' not in title
        assert '\r' not in title
        self.handle.write(f'>{title}\n')
        data = _get_seq_string(record)
        assert '\n' not in data
        assert '\r' not in data
        if self.wrap:
            for i in range(0, len(data), self.wrap):
                self.handle.write(data[i:i + self.wrap] + '\n')
        else:
            self.handle.write(data + '\n')