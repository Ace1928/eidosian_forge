from __future__ import annotations
import errno
import os
import sys
from contextlib import contextmanager
from typing import IO, Iterator, TextIO

    Ensure that the FD for `io` is set to blocking in here.
    