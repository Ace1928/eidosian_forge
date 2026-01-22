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
class LimitRequestFieldSize(Setting):
    name = 'limit_request_field_size'
    section = 'Security'
    cli = ['--limit-request-field_size']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 8190
    desc = '        Limit the allowed size of an HTTP request header field.\n\n        Value is a positive number or 0. Setting it to 0 will allow unlimited\n        header field sizes.\n\n        .. warning::\n           Setting this parameter to a very high or unlimited value can open\n           up for DDOS attacks.\n        '