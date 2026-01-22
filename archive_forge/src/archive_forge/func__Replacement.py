from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def _Replacement(match):
    """Returns one replacement string for LazyFormat re.sub()."""
    prefix = match.group(1)[1:]
    name = match.group(2)
    suffix = match.group(3)[1:]
    if prefix and suffix:
        return prefix + name + suffix
    value = kwargs.get(name)
    if value is None:
        return match.group(0)
    if callable(value):
        value = value()
    return prefix + LazyFormat(value, **kwargs) + suffix