import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_id(record, line):
    cols = line[5:].split()
    if len(cols) == 5:
        record.entry_name = cols[0]
        record.data_class = cols[1].rstrip(';')
        record.molecule_type = cols[2].rstrip(';')
        record.sequence_length = int(cols[3])
    elif len(cols) == 4:
        record.entry_name = cols[0]
        record.data_class = cols[1].rstrip(';')
        record.molecule_type = None
        record.sequence_length = int(cols[2])
    else:
        raise SwissProtParserError('ID line has unrecognised format', line=line)
    allowed = ('STANDARD', 'PRELIMINARY', 'IPI', 'Reviewed', 'Unreviewed')
    if record.data_class not in allowed:
        message = f'Unrecognized data class {record.data_class!r}'
        raise SwissProtParserError(message, line=line)
    if record.molecule_type not in (None, 'PRT'):
        message = f'Unrecognized molecule type {record.molecule_type!r}'
        raise SwissProtParserError(message, line=line)