from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
def track_usage(tool_id, tags):
    """No usage tracking for external library.

  Args:
    tool_id: A string identifier for tool to be tracked.
    tags: list of string tags that will be added to the tracking.
  """
    del tool_id, tags