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
class PreloadApp(Setting):
    name = 'preload_app'
    section = 'Server Mechanics'
    cli = ['--preload']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '        Load application code before the worker processes are forked.\n\n        By preloading an application you can save some RAM resources as well as\n        speed up server boot times. Although, if you defer application loading\n        to each worker process, you can reload your application code easily by\n        restarting workers.\n        '