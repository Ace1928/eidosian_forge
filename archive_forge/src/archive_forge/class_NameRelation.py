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
class NameRelation(dns.enum.IntEnum):
    """Name relation result from fullcompare()."""
    NONE = 0
    SUPERDOMAIN = 1
    SUBDOMAIN = 2
    EQUAL = 3
    COMMONANCESTOR = 4

    @classmethod
    def _maximum(cls):
        return cls.COMMONANCESTOR

    @classmethod
    def _short_name(cls):
        return cls.__name__