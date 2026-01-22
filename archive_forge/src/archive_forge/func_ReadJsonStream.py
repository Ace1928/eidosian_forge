from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import json
import six
def ReadJsonStream(file_obj):
    """Read the events from the skaffold event stream.

  Args:
    file_obj: A File object.

  Yields:
    Event dicts from the JSON payloads.
  """
    for line in _ReadStreamingLines(file_obj):
        if not line:
            continue
        yield json.loads(six.ensure_str(line))