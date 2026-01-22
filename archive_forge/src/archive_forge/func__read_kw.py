import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_kw(record, value):
    for val in value.rstrip(';.').split('; '):
        if val.endswith('}'):
            val = val.rsplit('{', 1)[0]
        record.keywords.append(val.strip())