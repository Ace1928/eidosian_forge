import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_rc(reference, value):
    cols = value.split(';')
    if value[-1] == ';':
        unread = ''
    else:
        cols, unread = (cols[:-1], cols[-1])
    for col in cols:
        if not col:
            return
        i = col.find('=')
        if i >= 0:
            token, text = (col[:i], col[i + 1:])
            comment = (token.lstrip(), text)
            reference.comments.append(comment)
        else:
            comment = reference.comments[-1]
            comment = f'{comment} {col}'
            reference.comments[-1] = comment
    return unread