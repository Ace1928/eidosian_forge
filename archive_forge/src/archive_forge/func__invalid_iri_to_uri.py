from __future__ import annotations
import codecs
import re
import typing as t
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from .datastructures import iter_multi_items
def _invalid_iri_to_uri(iri: str) -> str:
    """The URL scheme ``itms-services://`` must contain the ``//`` even though it does
    not have a host component. There may be other invalid schemes as well. Currently,
    responses will always call ``iri_to_uri`` on the redirect ``Location`` header, which
    removes the ``//``. For now, if the IRI only contains ASCII and does not contain
    spaces, pass it on as-is. In Werkzeug 3.0, this should become a
    ``response.process_location`` flag.

    :meta private:
    """
    try:
        iri.encode('ascii')
    except UnicodeError:
        pass
    else:
        if len(iri.split(None, 1)) == 1:
            return iri
    return iri_to_uri(iri)