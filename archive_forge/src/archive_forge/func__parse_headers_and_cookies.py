import argparse
import warnings
from http.cookies import SimpleCookie
from shlex import split
from urllib.parse import urlparse
from w3lib.http import basic_auth_header
def _parse_headers_and_cookies(parsed_args):
    headers = []
    cookies = {}
    for header in parsed_args.headers or ():
        name, val = header.split(':', 1)
        name = name.strip()
        val = val.strip()
        if name.title() == 'Cookie':
            for name, morsel in SimpleCookie(val).items():
                cookies[name] = morsel.value
        else:
            headers.append((name, val))
    if parsed_args.auth:
        user, password = parsed_args.auth.split(':', 1)
        headers.append(('Authorization', basic_auth_header(user, password)))
    return (headers, cookies)