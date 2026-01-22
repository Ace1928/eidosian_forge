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
class MaxRequests(Setting):
    name = 'max_requests'
    section = 'Worker Processes'
    cli = ['--max-requests']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 0
    desc = '        The maximum number of requests a worker will process before restarting.\n\n        Any value greater than zero will limit the number of requests a worker\n        will process before automatically restarting. This is a simple method\n        to help limit the damage of memory leaks.\n\n        If this is set to zero (the default) then the automatic worker\n        restarts are disabled.\n        '