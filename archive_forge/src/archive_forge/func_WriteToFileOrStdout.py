from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def WriteToFileOrStdout(path, content, overwrite=True, binary=False, private=False, create_path=False):
    """Writes content to the specified file or stdout if path is '-'.

  Args:
    path: str, The path of the file to write.
    content: str, The content to write to the file.
    overwrite: bool, Whether or not to overwrite the file if it exists.
    binary: bool, True to open the file in binary mode.
    private: bool, Whether to write the file in private mode.
    create_path: bool, True to create intermediate directories, if needed.

  Raises:
    Error: If the file cannot be written.
  """
    if path == '-':
        if binary:
            files.WriteStreamBytes(sys.stdout, content)
        else:
            out.write(content)
    elif binary:
        files.WriteBinaryFileContents(path, content, overwrite=overwrite, private=private, create_path=create_path)
    else:
        files.WriteFileContents(path, content, overwrite=overwrite, private=private, create_path=create_path)