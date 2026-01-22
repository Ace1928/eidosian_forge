from __future__ import annotations
import collections.abc
import io
import itertools
import os
import re
import string
from qiskit.circuit import (
from qiskit.circuit.tools import pi_check
from .exceptions import QASM2ExportError
def _escape_name(name: str, prefix: str) -> str:
    """Returns a valid OpenQASM 2.0 identifier, using `prefix` as a prefix if necessary.  `prefix`
    must itself be a valid identifier."""
    escaped_name = re.sub('\\W', '_', name, flags=re.ASCII)
    if not escaped_name or escaped_name[0] not in string.ascii_lowercase or escaped_name in _RESERVED:
        escaped_name = prefix + escaped_name
    return escaped_name