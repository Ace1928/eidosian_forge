from __future__ import annotations
from collections import defaultdict
import re
from mdurl._url import URL
def url_parse(url: URL | str, *, slashes_denote_host: bool=False) -> URL:
    if isinstance(url, URL):
        return url
    u = MutableURL()
    u.parse(url, slashes_denote_host)
    return URL(u.protocol, u.slashes, u.auth, u.port, u.hostname, u.hash, u.search, u.pathname)