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
def FormatRequiredUserAction(s):
    """Formats an action a user must initiate to complete a command.

  Some actions can't be prompted or initiated by gcloud itself, but they must
  be completed to accomplish the task requested of gcloud; the canonical example
  is that after installation or update, the user must restart their shell for
  all aspects of the update to take effect. Unlike most console output, such
  instructions need to be highlighted in some way. Using this function ensures
  that all such instances are highlighted the *same* way.

  Args:
    s: str, The message to format. It shouldn't begin or end with newlines.

  Returns:
    str, The formatted message. This should be printed starting on its own
      line, and followed by a newline.
  """
    with _NarrowWrap(4) as wrapper:
        separator = '\n==> '
        prefix = separator
        screen_reader = properties.VALUES.accessibility.screen_reader.GetBool()
        if screen_reader:
            separator = '\n '
            prefix = '\n'
        return prefix + separator.join(wrapper.wrap(s)) + '\n'