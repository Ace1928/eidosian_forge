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
class SyslogFacility(Setting):
    name = 'syslog_facility'
    section = 'Logging'
    cli = ['--log-syslog-facility']
    meta = 'SYSLOG_FACILITY'
    validator = validate_string
    default = 'user'
    desc = '    Syslog facility name\n    '