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
def _StacktraceEntryReplacement(entry):
    """Used in re.sub to format a stacktrace entry to make it more compact.

  File "qux.py", line 13, in run      ===>      qux.py:13
    foo = math.sqrt(bar) / foo                   foo = math.sqrt(bar)...

  Args:
    entry: re.MatchObject, the original unformatted stacktrace entry

  Returns:
    str, the formatted stacktrace entry
  """
    filename = entry.group('file')
    line_no = entry.group('line')
    code_snippet = entry.group('code_snippet')
    formatted_code_snippet = code_snippet.strip()[:MAX_CODE_SNIPPET_LENGTH]
    if len(code_snippet) > MAX_CODE_SNIPPET_LENGTH:
        formatted_code_snippet += '...'
    formatted_entry = '{0}:{1}\n {2}\n'.format(filename, line_no, formatted_code_snippet)
    return formatted_entry