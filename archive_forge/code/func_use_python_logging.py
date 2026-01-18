from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def use_python_logging(quiet=False):
    """Uses the python implementation of the logging code.

  Args:
    quiet: No logging message about switching logging type.
  """
    get_absl_handler().activate_python_handler()
    if not quiet:
        info('Restoring pure python logging')