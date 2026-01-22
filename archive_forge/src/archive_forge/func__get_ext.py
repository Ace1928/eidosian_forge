from __future__ import annotations
from calendar import timegm
from os.path import splitext
from time import altzone, gmtime, localtime, time, timezone
from typing import (
from urllib.parse import quote, urlsplit, urlunsplit
import rdflib.graph  # avoid circular dependency
import rdflib.namespace
import rdflib.term
from rdflib.compat import sign
def _get_ext(fpath: str, lower: bool=True) -> str:
    """
    Gets the file extension from a file(path); stripped of leading '.' and in
    lower case. Examples:

        >>> _get_ext("path/to/file.txt")
        'txt'
        >>> _get_ext("OTHER.PDF")
        'pdf'
        >>> _get_ext("noext")
        ''
        >>> _get_ext(".rdf")
        'rdf'
    """
    ext = splitext(fpath)[-1]
    if ext == '' and fpath.startswith('.'):
        ext = fpath
    if lower:
        ext = ext.lower()
    if ext.startswith('.'):
        ext = ext[1:]
    return ext