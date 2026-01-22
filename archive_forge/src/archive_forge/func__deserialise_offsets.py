import os
from ... import urlutils
from . import request
def _deserialise_offsets(self, text):
    offsets = []
    for line in text.split(b'\n'):
        if not line:
            continue
        start, length = line.split(b',')
        offsets.append((int(start), int(length)))
    return offsets