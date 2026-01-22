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
class Pidfile(Setting):
    name = 'pidfile'
    section = 'Server Mechanics'
    cli = ['-p', '--pid']
    meta = 'FILE'
    validator = validate_string
    default = None
    desc = '        A filename to use for the PID file.\n\n        If not set, no PID file will be written.\n        '