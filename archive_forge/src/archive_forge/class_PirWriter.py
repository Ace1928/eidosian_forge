from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class PirWriter(SequenceWriter):
    """Class to write PIR format files."""

    def __init__(self, handle, wrap=60, record2title=None, code=None):
        """Create a PIR writer.

        Arguments:
         - handle - Handle to an output file, e.g. as returned
           by open(filename, "w")
         - wrap - Optional line length used to wrap sequence lines.
           Defaults to wrapping the sequence at 60 characters
           Use zero (or None) for no wrapping, giving a single
           long line for the sequence.
         - record2title - Optional function to return the text to be
           used for the title line of each record.  By default
           a combination of the record.id, record.name and
           record.description is used.
         - code - Optional sequence code must be one of P1, F1,
           D1, DL, DC, RL, RC, N3 and XX. By default None is used,
           which means auto detection based on the molecule type
           in the record annotation.

        You can either use::

            handle = open(filename, "w")
            writer = PirWriter(handle)
            writer.write_file(myRecords)
            handle.close()

        Or, follow the sequential file writer system, for example::

            handle = open(filename, "w")
            writer = PirWriter(handle)
            writer.write_header() # does nothing for PIR files
            ...
            Multiple writer.write_record() and/or writer.write_records() calls
            ...
            writer.write_footer() # does nothing for PIR files
            handle.close()

        """
        super().__init__(handle)
        self.wrap = None
        if wrap:
            if wrap < 1:
                raise ValueError('wrap should be None, 0, or a positive integer')
        self.wrap = wrap
        self.record2title = record2title
        self.code = code

    def write_record(self, record):
        """Write a single PIR record to the file."""
        if self.record2title:
            title = self.clean(self.record2title(record))
        else:
            title = self.clean(record.id)
        if record.name and record.description:
            description = self.clean(record.name + ' - ' + record.description)
        elif record.name and (not record.description):
            description = self.clean(record.name)
        else:
            description = self.clean(record.description)
        if self.code:
            code = self.code
        else:
            molecule_type = record.annotations.get('molecule_type')
            if molecule_type is None:
                code = 'XX'
            elif 'DNA' in molecule_type:
                code = 'D1'
            elif 'RNA' in molecule_type:
                code = 'RL'
            elif 'protein' in molecule_type:
                code = 'P1'
            else:
                code = 'XX'
        if code not in _pir_mol_type:
            raise TypeError('Sequence code must be one of ' + _pir_mol_type.keys() + '.')
        assert '\n' not in title
        assert '\r' not in description
        self.handle.write(f'>{code};{title}\n{description}\n')
        data = _get_seq_string(record)
        assert '\n' not in data
        assert '\r' not in data
        if self.wrap:
            line = ''
            for i in range(0, len(data), self.wrap):
                line += data[i:i + self.wrap] + '\n'
            line = line[:-1] + '*\n'
            self.handle.write(line)
        else:
            self.handle.write(data + '*\n')