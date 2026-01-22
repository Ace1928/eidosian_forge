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
def _iri2uri(iri: str) -> str:
    """
    Prior art:

    * `iri_to_uri from Werkzeug <https://github.com/pallets/werkzeug/blob/92c6380248c7272ee668e1f8bbd80447027ccce2/src/werkzeug/urls.py#L926-L931>`_

    >>> _iri2uri("https://dbpedia.org/resource/Almería")
    'https://dbpedia.org/resource/Almer%C3%ADa'
    """
    parts = urlsplit(iri)
    scheme, netloc, path, query, fragment = parts
    if scheme not in ['http', 'https']:
        return iri
    path = quote(path, safe=_PATH_SAFE_CHARS)
    query = quote(query, safe=_QUERY_SAFE_CHARS)
    fragment = quote(fragment, safe=_QUERY_SAFE_CHARS)
    if parts.hostname:
        netloc = parts.hostname.encode('idna').decode('ascii')
    else:
        netloc = ''
    if ':' in netloc:
        netloc = f'[{netloc}]'
    if parts.port:
        netloc = f'{netloc}:{parts.port}'
    if parts.username:
        auth = quote(parts.username, safe=_USERNAME_SAFE_CHARS)
        if parts.password:
            pass_quoted = quote(parts.password, safe=_USERNAME_SAFE_CHARS)
            auth = f'{auth}:{pass_quoted}'
        netloc = f'{auth}@{netloc}'
    uri = urlunsplit((scheme, netloc, path, query, fragment))
    if iri.endswith('#') and (not uri.endswith('#')):
        uri += '#'
    return uri