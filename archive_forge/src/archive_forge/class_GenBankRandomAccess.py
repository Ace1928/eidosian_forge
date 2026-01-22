import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class GenBankRandomAccess(SequentialSeqFileRandomAccess):
    """Indexed dictionary like access to a GenBank file."""

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        marker_re = self._marker_re
        accession_marker = b'ACCESSION '
        version_marker = b'VERSION '
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
        while marker_re.match(line):
            try:
                key = line[5:].split(None, 1)[0]
            except ValueError:
                key = None
            length = len(line)
            while True:
                end_offset = handle.tell()
                line = handle.readline()
                if marker_re.match(line) or not line:
                    if not key:
                        raise ValueError('Did not find usable ACCESSION/VERSION/LOCUS lines')
                    yield (key.decode(), start_offset, length)
                    start_offset = end_offset
                    break
                elif line.startswith(accession_marker):
                    try:
                        key = line.rstrip().split()[1]
                    except IndexError:
                        pass
                elif line.startswith(version_marker):
                    try:
                        version_id = line.rstrip().split()[1]
                        if version_id.count(b'.') == 1 and version_id.split(b'.')[1].isdigit():
                            key = version_id
                    except IndexError:
                        pass
                length += len(line)
        assert not line, repr(line)