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
class AccessLogFormat(Setting):
    name = 'access_log_format'
    section = 'Logging'
    cli = ['--access-logformat']
    meta = 'STRING'
    validator = validate_string
    default = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
    desc = "        The access log format.\n\n        ===========  ===========\n        Identifier   Description\n        ===========  ===========\n        h            remote address\n        l            ``'-'``\n        u            user name\n        t            date of the request\n        r            status line (e.g. ``GET / HTTP/1.1``)\n        m            request method\n        U            URL path without query string\n        q            query string\n        H            protocol\n        s            status\n        B            response length\n        b            response length or ``'-'`` (CLF format)\n        f            referer\n        a            user agent\n        T            request time in seconds\n        M            request time in milliseconds\n        D            request time in microseconds\n        L            request time in decimal seconds\n        p            process ID\n        {header}i    request header\n        {header}o    response header\n        {variable}e  environment variable\n        ===========  ===========\n\n        Use lowercase for header and environment variable names, and put\n        ``{...}x`` names inside ``%(...)s``. For example::\n\n            %({x-forwarded-for}i)s\n        "