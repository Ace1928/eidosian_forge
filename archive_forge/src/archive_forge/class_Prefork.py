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
class Prefork(Setting):
    name = 'pre_fork'
    section = 'Server Hooks'
    validator = validate_callable(2)
    type = callable

    def pre_fork(server, worker):
        pass
    default = staticmethod(pre_fork)
    desc = '        Called just before a worker is forked.\n\n        The callable needs to accept two instance variables for the Arbiter and\n        new Worker.\n        '