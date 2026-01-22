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
class Chdir(Setting):
    name = 'chdir'
    section = 'Server Mechanics'
    cli = ['--chdir']
    validator = validate_chdir
    default = util.getcwd()
    default_doc = "``'.'``"
    desc = '        Change directory to specified directory before loading apps.\n        '