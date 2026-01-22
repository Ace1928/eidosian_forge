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
class LogConfigDict(Setting):
    name = 'logconfig_dict'
    section = 'Logging'
    validator = validate_dict
    default = {}
    desc = "    The log config dictionary to use, using the standard Python\n    logging module's dictionary configuration format. This option\n    takes precedence over the :ref:`logconfig` and :ref:`logConfigJson` options,\n    which uses the older file configuration format and JSON\n    respectively.\n\n    Format: https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig\n\n    For more context you can look at the default configuration dictionary for logging,\n    which can be found at ``gunicorn.glogging.CONFIG_DEFAULTS``.\n\n    .. versionadded:: 19.8\n    "