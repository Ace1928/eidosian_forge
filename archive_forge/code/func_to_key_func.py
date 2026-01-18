import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def to_key_func(i: int) -> _ToKeyFunc:

    def to_key(triple: Tuple[bytes, bytes, bytes], context: bytes) -> bytes:
        """Takes a string; returns key"""
        return '^'.encode('latin-1').join((context, triple[i % 3], triple[(i + 1) % 3], triple[(i + 2) % 3], ''.encode('latin-1')))
    return to_key