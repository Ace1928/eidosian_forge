from __future__ import absolute_import
from .snappy import (
Tries to guess a compression format for the given input file by it's
    header.
    :return: tuple of decompression method and a chunk that was taken from the
        input for format detection.
    