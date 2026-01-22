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
class EnableStdioInheritance(Setting):
    name = 'enable_stdio_inheritance'
    section = 'Logging'
    cli = ['-R', '--enable-stdio-inheritance']
    validator = validate_bool
    default = False
    action = 'store_true'
    desc = '    Enable stdio inheritance.\n\n    Enable inheritance for stdio file descriptors in daemon mode.\n\n    Note: To disable the Python stdout buffering, you can to set the user\n    environment variable ``PYTHONUNBUFFERED`` .\n    '