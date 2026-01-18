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
def validate_list_string(val):
    if not val:
        return []
    if isinstance(val, str):
        val = [val]
    return [validate_string(v) for v in val]