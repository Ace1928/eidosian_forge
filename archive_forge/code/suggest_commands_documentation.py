from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import re
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
Return suggested commands containing input command words.

  Args:
    command_words: List of input command words.

  Returns:
    [command]: A list of canonical command strings with 'gcloud' prepended. Only
      commands whose scores have a ratio of at least MIN_RATIO against the top
      score are returned. At most MAX_SUGGESTIONS command strings are returned.
      If many commands from the same group are being suggested, then the common
      groups are suggested instead.
  