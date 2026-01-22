from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _uniq_stable(elems: Iterable) -> list:
    """uniq_stable(elems) -> list

    Return from an iterable, a list of all the unique elements in the input,
    maintaining the order in which they first appear.
    """
    seen = set()
    value = []
    for x in elems:
        if x not in seen:
            value.append(x)
            seen.add(x)
    return value