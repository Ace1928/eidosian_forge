from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def _validate_labels(labels):
    """Check for empty labels in the middle of a label sequence,
    labels that are too long, and for too many labels.

    Raises ``dns.name.NameTooLong`` if the name as a whole is too long.

    Raises ``dns.name.EmptyLabel`` if a label is empty (i.e. the root
    label) and appears in a position other than the end of the label
    sequence

    """
    l = len(labels)
    total = 0
    i = -1
    j = 0
    for label in labels:
        ll = len(label)
        total += ll + 1
        if ll > 63:
            raise LabelTooLong
        if i < 0 and label == b'':
            i = j
        j += 1
    if total > 255:
        raise NameTooLong
    if i >= 0 and i != l - 1:
        raise EmptyLabel