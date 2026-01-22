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
class SSLVersion(Setting):
    name = 'ssl_version'
    section = 'SSL'
    cli = ['--ssl-version']
    validator = validate_ssl_version
    if hasattr(ssl, 'PROTOCOL_TLS'):
        default = ssl.PROTOCOL_TLS
    else:
        default = ssl.PROTOCOL_SSLv23
    default = ssl.PROTOCOL_SSLv23
    desc = "    SSL version to use (see stdlib ssl module's).\n\n    .. deprecated:: 20.2\n       The option is deprecated and it is currently ignored. Use :ref:`ssl-context` instead.\n\n    ============= ============\n    --ssl-version Description\n    ============= ============\n    SSLv3         SSLv3 is not-secure and is strongly discouraged.\n    SSLv23        Alias for TLS. Deprecated in Python 3.6, use TLS.\n    TLS           Negotiate highest possible version between client/server.\n                  Can yield SSL. (Python 3.6+)\n    TLSv1         TLS 1.0\n    TLSv1_1       TLS 1.1 (Python 3.4+)\n    TLSv1_2       TLS 1.2 (Python 3.4+)\n    TLS_SERVER    Auto-negotiate the highest protocol version like TLS,\n                  but only support server-side SSLSocket connections.\n                  (Python 3.6+)\n    ============= ============\n\n    .. versionchanged:: 19.7\n       The default value has been changed from ``ssl.PROTOCOL_TLSv1`` to\n       ``ssl.PROTOCOL_SSLv23``.\n    .. versionchanged:: 20.0\n       This setting now accepts string names based on ``ssl.PROTOCOL_``\n       constants.\n    .. versionchanged:: 20.0.1\n       The default value has been changed from ``ssl.PROTOCOL_SSLv23`` to\n       ``ssl.PROTOCOL_TLS`` when Python >= 3.6 .\n    "