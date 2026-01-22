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
class DisableRedirectAccessToSyslog(Setting):
    name = 'disable_redirect_access_to_syslog'
    section = 'Logging'
    cli = ['--disable-redirect-access-to-syslog']
    validator = validate_bool
    action = 'store_true'
    default = False
    desc = '    Disable redirect access logs to syslog.\n\n    .. versionadded:: 19.8\n    '