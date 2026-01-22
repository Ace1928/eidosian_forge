import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
class FilePathOrStdinContents(object):
    """Creates an argparse type that stores a file path or the contents of stdin.

  This is similar to FileContents above but only reads content from stdin,
  otherwise just stores the file/directory path.

  Attributes:
    binary: bool, If True, the contents of the file will be returned as bytes.

  Returns:
    A function that accepts a filename, or "-" representing that stdin should be
    used as input.
  """

    def __init__(self, binary=False):
        self.binary = binary

    def __call__(self, name):
        """Return the contents of stdin or the filepath specified.

    If name is "-", stdin is read until EOF. Otherwise, the named file path is
    returned.

    Args:
      name: str, The file name, or '-' to indicate stdin.

    Returns:
      The contents of stdin or the file path.

    Raises:
      ArgumentTypeError: If stdin cannot be read or is too large.
    """
        try:
            if name == '-':
                return console_io.ReadFromFileOrStdin(name, binary=self.binary)
            return files.ExpandHomeAndVars(name)
        except files.Error as e:
            raise ArgumentTypeError(e)