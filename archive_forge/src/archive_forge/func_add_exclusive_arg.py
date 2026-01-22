import collections
import getpass
import inspect
import os
import sys
import textwrap
import decorator
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from zunclient.i18n import _
def add_exclusive_arg(func, group_name, required, *args, **kwargs):
    """Bind CLI mutally exclusive arguments to a shell.py `do_foo` function."""
    if not hasattr(func, 'exclusive_args'):
        func.exclusive_args = collections.defaultdict(list)
        func.exclusive_args['__required__'] = collections.defaultdict(bool)
    if (args, kwargs) not in func.exclusive_args[group_name]:
        func.exclusive_args[group_name].insert(0, (args, kwargs))
        if required is not None:
            func.exclusive_args['__required__'][group_name] = required