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
def _pad_to_max_label(label, suffix_labels):
    length = len(label)
    remaining = 255 - _wire_length(suffix_labels) - length - 1
    if remaining <= 0:
        return label
    needed = min(63 - length, remaining)
    return label + _MAXIMAL_OCTET * needed