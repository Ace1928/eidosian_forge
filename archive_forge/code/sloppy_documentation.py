from __future__ import annotations
import codecs
from encodings import normalize_encoding
import sys

    Take a codec name, and return a 'sloppy' version of that codec that can
    encode and decode the unassigned bytes in that encoding.

    Single-byte encodings in the standard library are defined using some
    boilerplate classes surrounding the functions that do the actual work,
    `codecs.charmap_decode` and `charmap_encode`. This function, given an
    encoding name, *defines* those boilerplate classes.
    