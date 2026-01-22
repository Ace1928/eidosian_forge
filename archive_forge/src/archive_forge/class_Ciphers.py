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
class Ciphers(Setting):
    name = 'ciphers'
    section = 'SSL'
    cli = ['--ciphers']
    validator = validate_string
    default = None
    desc = "    SSL Cipher suite to use, in the format of an OpenSSL cipher list.\n\n    By default we use the default cipher list from Python's ``ssl`` module,\n    which contains ciphers considered strong at the time of each Python\n    release.\n\n    As a recommended alternative, the Open Web App Security Project (OWASP)\n    offers `a vetted set of strong cipher strings rated A+ to C-\n    <https://www.owasp.org/index.php/TLS_Cipher_String_Cheat_Sheet>`_.\n    OWASP provides details on user-agent compatibility at each security level.\n\n    See the `OpenSSL Cipher List Format Documentation\n    <https://www.openssl.org/docs/manmaster/man1/ciphers.html#CIPHER-LIST-FORMAT>`_\n    for details on the format of an OpenSSL cipher list.\n    "