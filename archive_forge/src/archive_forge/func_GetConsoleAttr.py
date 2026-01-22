from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
def GetConsoleAttr(encoding=None, term=None, reset=False):
    """Gets the console attribute state.

  If this is the first call or reset is True or encoding is not None and does
  not match the current encoding or out is not None and does not match the
  current out then the state is (re)initialized. Otherwise the current state
  is returned.

  This call associates the out file stream with the console. All console related
  output should go to the same stream.

  Args:
    encoding: Encoding override.
      ascii -- ASCII. This is the default.
      utf8 -- UTF-8 unicode.
      win -- Windows code page 437.
    term: Terminal override. Replaces the value of ENV['TERM'].
    reset: Force re-initialization if True.

  Returns:
    The global ConsoleAttr state object.
  """
    attr = ConsoleAttr._CONSOLE_ATTR_STATE
    if not reset:
        if not attr:
            reset = True
        elif encoding and encoding != attr.GetEncoding():
            reset = True
    if reset:
        attr = ConsoleAttr(encoding=encoding, term=term)
        ConsoleAttr._CONSOLE_ATTR_STATE = attr
    return attr