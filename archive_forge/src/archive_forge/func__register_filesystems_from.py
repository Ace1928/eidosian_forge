from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
def _register_filesystems_from(fsspec_registry, only_available):
    """Register each fsspec provider from this registry as remote source."""
    for protocol, spec in fsspec_registry.items():
        missing_deps = isinstance(spec, dict) and 'err' in spec
        if missing_deps and only_available:
            continue
        has_reader = get_reader(protocol)
        if not missing_deps or has_reader is None:
            register_reader(protocol, RemoteSource)
        has_writer = get_writer(protocol)
        if not missing_deps or has_writer is None:
            register_writer(protocol, RemoteSource)