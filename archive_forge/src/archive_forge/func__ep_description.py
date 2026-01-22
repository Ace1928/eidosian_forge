import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def _ep_description(self, ep, pad_name=None):
    name = ep.name
    if pad_name is not None:
        name = name + ' ' * (pad_name - len(name))
    dest = ep.module_name
    if ep.attrs:
        dest = dest + ':' + '.'.join(ep.attrs)
    return '%s = %s' % (name, dest)