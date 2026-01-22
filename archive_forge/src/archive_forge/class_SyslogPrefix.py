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
class SyslogPrefix(Setting):
    name = 'syslog_prefix'
    section = 'Logging'
    cli = ['--log-syslog-prefix']
    meta = 'SYSLOG_PREFIX'
    validator = validate_string
    default = None
    desc = '    Makes Gunicorn use the parameter as program-name in the syslog entries.\n\n    All entries will be prefixed by ``gunicorn.<prefix>``. By default the\n    program name is the name of the process.\n    '