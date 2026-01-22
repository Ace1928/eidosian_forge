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
def _absolute_successor(name: Name, origin: Name, prefix_ok: bool) -> Name:
    if prefix_ok:
        try:
            return _SUCCESSOR_PREFIX.concatenate(name)
        except NameTooLong:
            pass
    while name != origin:
        least_significant_label = name[0]
        if len(least_significant_label) < 63:
            new_labels = [least_significant_label + _MINIMAL_OCTET]
            new_labels.extend(name.labels[1:])
            try:
                return dns.name.Name(new_labels)
            except dns.name.NameTooLong:
                pass
        octets = bytearray(least_significant_label)
        for i in range(len(octets) - 1, -1, -1):
            octet = octets[i]
            if octet == _MAXIMAL_OCTET_VALUE:
                continue
            if octet == _AT_SIGN_VALUE:
                octet = _LEFT_SQUARE_BRACKET_VALUE
            else:
                octet += 1
            octets[i] = octet
            new_labels = [bytes(octets[:i + 1])]
            new_labels.extend(name.labels[1:])
            return Name(new_labels)
        name = name.parent()
    return origin