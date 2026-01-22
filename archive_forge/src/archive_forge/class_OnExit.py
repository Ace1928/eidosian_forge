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
class OnExit(Setting):
    name = 'on_exit'
    section = 'Server Hooks'
    validator = validate_callable(1)

    def on_exit(server):
        pass
    default = staticmethod(on_exit)
    desc = '        Called just before exiting Gunicorn.\n\n        The callable needs to accept a single instance variable for the Arbiter.\n        '