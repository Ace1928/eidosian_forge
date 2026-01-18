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
def validate_chdir(val):
    val = validate_string(val)
    path = os.path.abspath(os.path.normpath(os.path.join(util.getcwd(), val)))
    if not os.path.exists(path):
        raise ConfigError("can't chdir to %r" % val)
    return path