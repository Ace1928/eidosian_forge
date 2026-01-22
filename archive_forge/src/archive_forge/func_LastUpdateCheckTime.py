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
def LastUpdateCheckTime(self):
    """Gets the time of the last update check as seconds since the epoch.

    Returns:
      int, The time of the last update check in seconds since the epoch.
    """
    return self._data.last_update_check_time