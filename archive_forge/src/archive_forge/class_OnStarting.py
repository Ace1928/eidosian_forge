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
class OnStarting(Setting):
    name = 'on_starting'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def on_starting(server):
        pass
    default = staticmethod(on_starting)
    desc = '        Called just before the master process is initialized.\n\n        The callable needs to accept a single instance variable for the Arbiter.\n        '