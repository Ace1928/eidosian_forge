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
class Postfork(Setting):
    name = 'post_fork'
    section = 'Server Hooks'
    validator = validate_callable(2)
    type = callable

    def post_fork(server, worker):
        pass
    default = staticmethod(post_fork)
    desc = '        Called just after a worker has been forked.\n\n        The callable needs to accept two instance variables for the Arbiter and\n        new Worker.\n        '