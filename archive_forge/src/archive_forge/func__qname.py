import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _qname(self, qname, uri=None):
    if uri is None:
        uri, tag = qname[1:].rsplit('}', 1) if qname[:1] == '{' else ('', qname)
    else:
        tag = qname
    prefixes_seen = set()
    for u, prefix in self._iter_namespaces(self._declared_ns_stack):
        if u == uri and prefix not in prefixes_seen:
            return (f'{prefix}:{tag}' if prefix else tag, tag, uri)
        prefixes_seen.add(prefix)
    if self._rewrite_prefixes:
        if uri in self._prefix_map:
            prefix = self._prefix_map[uri]
        else:
            prefix = self._prefix_map[uri] = f'n{len(self._prefix_map)}'
        self._declared_ns_stack[-1].append((uri, prefix))
        return (f'{prefix}:{tag}', tag, uri)
    if not uri and '' not in prefixes_seen:
        return (tag, tag, uri)
    for u, prefix in self._iter_namespaces(self._ns_stack):
        if u == uri:
            self._declared_ns_stack[-1].append((uri, prefix))
            return (f'{prefix}:{tag}' if prefix else tag, tag, uri)
    if not uri:
        return (tag, tag, uri)
    raise ValueError(f'Namespace "{uri}" is not declared in scope')