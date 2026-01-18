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
def validate_list_of_existing_files(val):
    return [validate_file_exists(v) for v in validate_list_string(val)]