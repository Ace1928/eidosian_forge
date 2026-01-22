import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class FastqRandomAccess(SeqFileRandomAccess):
    """Random access to a FASTQ file (any supported variant).

    With FASTQ the records all start with a "@" line, but so can quality lines.
    Note this will cope with line-wrapped FASTQ files.
    """

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        id = None
        start_offset = handle.tell()
        line = handle.readline()
        if not line:
            return
        if line[0:1] != b'@':
            raise ValueError(f'Problem with FASTQ @ line:\n{line!r}')
        while line:
            id = line[1:].rstrip().split(None, 1)[0]
            seq_len = 0
            length = len(line)
            while line:
                line = handle.readline()
                length += len(line)
                if line.startswith(b'+'):
                    break
                seq_len += len(line.strip())
            if not line:
                raise ValueError('Premature end of file in seq section')
            qual_len = 0
            while line:
                if seq_len == qual_len:
                    if seq_len == 0:
                        line = handle.readline()
                        if line.strip():
                            raise ValueError(f'Expected blank quality line, not {line!r}')
                        length += len(line)
                    end_offset = handle.tell()
                    line = handle.readline()
                    if line and line[0:1] != b'@':
                        raise ValueError(f'Problem with line {line!r}')
                    break
                else:
                    line = handle.readline()
                    qual_len += len(line.strip())
                    length += len(line)
            if seq_len != qual_len:
                raise ValueError('Problem with quality section')
            yield (id.decode(), start_offset, length)
            start_offset = end_offset

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        handle.seek(offset)
        line = handle.readline()
        data = line
        if line[0:1] != b'@':
            raise ValueError(f'Problem with FASTQ @ line:\n{line!r}')
        seq_len = 0
        while line:
            line = handle.readline()
            data += line
            if line.startswith(b'+'):
                break
            seq_len += len(line.strip())
        if not line:
            raise ValueError('Premature end of file in seq section')
        assert line[0:1] == b'+'
        qual_len = 0
        while line:
            if seq_len == qual_len:
                if seq_len == 0:
                    line = handle.readline()
                    if line.strip():
                        raise ValueError(f'Expected blank quality line, not {line!r}')
                    data += line
                line = handle.readline()
                if line and line[0:1] != b'@':
                    raise ValueError(f'Problem with line {line!r}')
                break
            else:
                line = handle.readline()
                data += line
                qual_len += len(line.strip())
        if seq_len != qual_len:
            raise ValueError('Problem with quality section')
        return data