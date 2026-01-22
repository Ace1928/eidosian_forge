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
class AccessLog(Setting):
    name = 'accesslog'
    section = 'Logging'
    cli = ['--access-logfile']
    meta = 'FILE'
    validator = validate_string
    default = None
    desc = "        The Access log file to write to.\n\n        ``'-'`` means log to stdout.\n        "