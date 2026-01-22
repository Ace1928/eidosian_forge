import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_ox(record, line):
    line = line.split('{')[0]
    if record.taxonomy_id:
        ids = line[5:].rstrip().rstrip(';')
    else:
        descr, ids = line[5:].rstrip().rstrip(';').split('=')
        assert descr == 'NCBI_TaxID', f'Unexpected taxonomy type {descr}'
    record.taxonomy_id.extend(ids.split(', '))