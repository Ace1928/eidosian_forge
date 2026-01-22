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
class Sendfile(Setting):
    name = 'sendfile'
    section = 'Server Mechanics'
    cli = ['--no-sendfile']
    validator = validate_bool
    action = 'store_const'
    const = False
    desc = '        Disables the use of ``sendfile()``.\n\n        If not set, the value of the ``SENDFILE`` environment variable is used\n        to enable or disable its usage.\n\n        .. versionadded:: 19.2\n        .. versionchanged:: 19.4\n           Swapped ``--sendfile`` with ``--no-sendfile`` to actually allow\n           disabling.\n        .. versionchanged:: 19.6\n           added support for the ``SENDFILE`` environment variable\n        '