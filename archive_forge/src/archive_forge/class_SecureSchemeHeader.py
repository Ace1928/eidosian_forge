import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class SecureSchemeHeader(Setting):
    name = 'secure_scheme_headers'
    section = 'Server Mechanics'
    validator = validate_dict
    default = {'X-FORWARDED-PROTOCOL': 'ssl', 'X-FORWARDED-PROTO': 'https', 'X-FORWARDED-SSL': 'on'}
    desc = "\n        A dictionary containing headers and values that the front-end proxy\n        uses to indicate HTTPS requests. If the source IP is permitted by\n        ``forwarded-allow-ips`` (below), *and* at least one request header matches\n        a key-value pair listed in this dictionary, then Gunicorn will set\n        ``wsgi.url_scheme`` to ``https``, so your application can tell that the\n        request is secure.\n\n        If the other headers listed in this dictionary are not present in the request, they will be ignored,\n        but if the other headers are present and do not match the provided values, then\n        the request will fail to parse. See the note below for more detailed examples of this behaviour.\n\n        The dictionary should map upper-case header names to exact string\n        values. The value comparisons are case-sensitive, unlike the header\n        names, so make sure they're exactly what your front-end proxy sends\n        when handling HTTPS requests.\n\n        It is important that your front-end proxy configuration ensures that\n        the headers defined here can not be passed directly from the client.\n        "