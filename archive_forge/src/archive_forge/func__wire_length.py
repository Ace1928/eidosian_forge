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
def _wire_length(labels):
    return functools.reduce(lambda v, x: v + len(x) + 1, labels, 0)