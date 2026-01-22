from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
import six
def SecondsSinceLastUpdateCheck(self):
    """Gets the number of seconds since we last did an update check.

    Returns:
      int, The amount of time in seconds.
    """
    return time.time() - self._data.last_update_check_time