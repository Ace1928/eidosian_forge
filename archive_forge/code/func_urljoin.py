from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def urljoin(base, url, allow_fragments=True):
    """Join a base URL and a possibly relative URL to form an absolute
    interpretation of the latter."""
    if not base:
        return url
    if not url:
        return base
    base, url, _coerce_result = _coerce_args(base, url)
    bscheme, bnetloc, bpath, bparams, bquery, bfragment = urlparse(base, '', allow_fragments)
    scheme, netloc, path, params, query, fragment = urlparse(url, bscheme, allow_fragments)
    if scheme != bscheme or scheme not in uses_relative:
        return _coerce_result(url)
    if scheme in uses_netloc:
        if netloc:
            return _coerce_result(urlunparse((scheme, netloc, path, params, query, fragment)))
        netloc = bnetloc
    if not path and (not params):
        path = bpath
        params = bparams
        if not query:
            query = bquery
        return _coerce_result(urlunparse((scheme, netloc, path, params, query, fragment)))
    base_parts = bpath.split('/')
    if base_parts[-1] != '':
        del base_parts[-1]
    if path[:1] == '/':
        segments = path.split('/')
    else:
        segments = base_parts + path.split('/')
        segments[1:-1] = filter(None, segments[1:-1])
    resolved_path = []
    for seg in segments:
        if seg == '..':
            try:
                resolved_path.pop()
            except IndexError:
                pass
        elif seg == '.':
            continue
        else:
            resolved_path.append(seg)
    if segments[-1] in ('.', '..'):
        resolved_path.append('')
    return _coerce_result(urlunparse((scheme, netloc, '/'.join(resolved_path) or '/', params, query, fragment)))