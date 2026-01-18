from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
def wants(self, exc):
    """Returns whether this handler wants to handle the exception or not.

    This base class returns True for all exceptions by default. Override in
    subclass if it wants to be more selective.

    Args:
      exc: Exception, the current exception.
    """
    del exc
    return True