import copy
import encodings.idna  # type: ignore
import functools
import struct
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import dns._features
import dns.enum
import dns.exception
import dns.immutable
import dns.wire
def _pad_to_max_name(name):
    needed = 255 - _wire_length(name.labels)
    new_labels = []
    while needed > 64:
        new_labels.append(_MAXIMAL_OCTET * 63)
        needed -= 64
    if needed >= 2:
        new_labels.append(_MAXIMAL_OCTET * (needed - 1))
    new_labels = list(reversed(new_labels))
    new_labels.extend(name.labels)
    return Name(new_labels)