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
class ReloadEngine(Setting):
    name = 'reload_engine'
    section = 'Debugging'
    cli = ['--reload-engine']
    meta = 'STRING'
    validator = validate_reload_engine
    default = 'auto'
    desc = "        The implementation that should be used to power :ref:`reload`.\n\n        Valid engines are:\n\n        * ``'auto'``\n        * ``'poll'``\n        * ``'inotify'`` (requires inotify)\n\n        .. versionadded:: 19.7\n        "