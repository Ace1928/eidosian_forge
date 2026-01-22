import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class UniprotRandomAccess(SequentialSeqFileRandomAccess):
    """Random access to a UniProt XML file."""

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        marker_re = self._marker_re
        start_acc_marker = b'<accession>'
        end_acc_marker = b'</accession>'
        end_entry_marker = b'</entry>'
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
        while marker_re.match(line):
            length = len(line)
            key = None
            while True:
                line = handle.readline()
                if key is None and start_acc_marker in line:
                    assert end_acc_marker in line, line
                    key = line[line.find(start_acc_marker) + 11:].split(b'<', 1)[0]
                    length += len(line)
                elif end_entry_marker in line:
                    length += line.find(end_entry_marker) + 8
                    end_offset = handle.tell() - len(line) + line.find(end_entry_marker) + 8
                    assert start_offset + length == end_offset
                    break
                elif marker_re.match(line) or not line:
                    raise ValueError("Didn't find end of record")
                else:
                    length += len(line)
            if not key:
                raise ValueError('Did not find <accession> line in bytes %i to %i' % (start_offset, start_offset + length))
            yield (key.decode(), start_offset, length)
            while not marker_re.match(line) and line:
                start_offset = handle.tell()
                line = handle.readline()
        assert not line, repr(line)

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        marker_re = self._marker_re
        end_entry_marker = b'</entry>'
        handle.seek(offset)
        data = [handle.readline()]
        while True:
            line = handle.readline()
            i = line.find(end_entry_marker)
            if i != -1:
                data.append(line[:i + 8])
                break
            if marker_re.match(line) or not line:
                raise ValueError("Didn't find end of record")
            data.append(line)
        return b''.join(data)

    def get(self, offset):
        """Return the SeqRecord starting at the given offset."""
        data = b'<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n        <uniprot xmlns="http://uniprot.org/uniprot"\n        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n        xsi:schemaLocation="http://uniprot.org/uniprot\n        http://www.uniprot.org/support/docs/uniprot.xsd">\n        ' + self.get_raw(offset) + b'</uniprot>'
        return next(SeqIO.UniprotIO.UniprotIterator(BytesIO(data)))