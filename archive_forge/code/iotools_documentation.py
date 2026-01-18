from __future__ import print_function, unicode_literals
import typing
import array
import io
from io import SEEK_CUR, SEEK_SET
from .mode import Mode
Iterate over the lines of a file.

    Implementation reads each char individually, which is not very
    efficient.

    Yields:
        str: a single line in the file.

    