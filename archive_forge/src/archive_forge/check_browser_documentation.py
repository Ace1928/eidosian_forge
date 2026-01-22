from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
Determines if a browser can be launched.

  Args:
    attempt_launch_browser: bool, True to launch a browser if it's possible in
      the user's environment; False to not even try.

  Returns:
    True if the tool should actually launch a browser, based on user preference
    and environment.
  