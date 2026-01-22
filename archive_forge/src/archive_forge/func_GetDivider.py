from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def GetDivider(text=''):
    """Return a console-width divider (ex: '======================' (etc.)).

  Supports messages (ex: '=== Messsage Here ===').

  Args:
    text: str, a message to display centered in the divider.

  Returns:
    str, the formatted divider
  """
    if text:
        text = ' ' + text + ' '
    width, _ = console_attr_os.GetTermSize()
    return text.center(width, '=')