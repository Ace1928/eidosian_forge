import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_cc(record, line):
    key, value = (line[5:8], line[9:].rstrip())
    if key == '-!-':
        record.comments.append(value)
    elif key == '   ':
        if not record.comments:
            record.comments.append(value)
        else:
            record.comments[-1] += ' ' + value