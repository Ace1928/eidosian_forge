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
def validate_pos_int(val):
    if not isinstance(val, int):
        val = int(val, 0)
    else:
        val = int(val)
    if val < 0:
        raise ValueError('Value must be positive: %s' % val)
    return val