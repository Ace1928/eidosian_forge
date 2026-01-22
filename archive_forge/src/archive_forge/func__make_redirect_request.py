from __future__ import annotations
import string
import sys
from typing import Dict
from urllib.error import HTTPError
from urllib.parse import quote as urlquote
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, urlopen
from urllib.response import addinfourl
def _make_redirect_request(request: Request, http_error: HTTPError) -> Request:
    """
    Create a new request object for a redirected request.

    The logic is based on `urllib.request.HTTPRedirectHandler` from `this commit <https://github.com/python/cpython/blob/b58bc8c2a9a316891a5ea1a0487aebfc86c2793a/Lib/urllib/request.py#L641-L751>_`.

    :param request: The original request that resulted in the redirect.
    :param http_error: The response to the original request that indicates a
        redirect should occur and contains the new location.
    :return: A new request object to the location indicated by the response.
    :raises HTTPError: the supplied ``http_error`` if the redirect request
        cannot be created.
    :raises ValueError: If the response code is `None`.
    :raises ValueError: If the response does not contain a ``Location`` header
        or the ``Location`` header is not a string.
    :raises HTTPError: If the scheme of the new location is not ``http``,
        ``https``, or ``ftp``.
    :raises HTTPError: If there are too many redirects or a redirect loop.
    """
    new_url = http_error.headers.get('Location')
    if new_url is None:
        raise http_error
    if not isinstance(new_url, str):
        raise ValueError(f'Location header {new_url!r} is not a string')
    new_url_parts = urlsplit(new_url)
    if new_url_parts.scheme not in ('http', 'https', 'ftp', ''):
        raise HTTPError(new_url, http_error.code, f'{http_error.reason} - Redirection to url {new_url!r} is not allowed', http_error.headers, http_error.fp)
    new_url = urlquote(new_url, encoding='iso-8859-1', safe=string.punctuation)
    new_url = urljoin(request.full_url, new_url)
    content_headers = ('content-length', 'content-type')
    newheaders = {k: v for k, v in request.headers.items() if k.lower() not in content_headers}
    new_request = Request(new_url, headers=newheaders, origin_req_host=request.origin_req_host, unverifiable=True)
    visited: Dict[str, int]
    if hasattr(request, 'redirect_dict'):
        visited = request.redirect_dict
        if visited.get(new_url, 0) >= HTTPRedirectHandler.max_repeats or len(visited) >= HTTPRedirectHandler.max_redirections:
            raise HTTPError(request.full_url, http_error.code, HTTPRedirectHandler.inf_msg + http_error.reason, http_error.headers, http_error.fp)
    else:
        visited = {}
        setattr(request, 'redirect_dict', visited)
    setattr(new_request, 'redirect_dict', visited)
    visited[new_url] = visited.get(new_url, 0) + 1
    return new_request