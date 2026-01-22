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
def _CleanUpLastNagTimes(self):
    """Clean the map holding the last nag times for each notification.

    If a notification is no longer activate, it is removed from the map.  Any
    notifications that are still activated have their last nag times preserved.
    """
    activated_ids = [n.id for n in self._data.notifications]
    self._data.last_nag_times = dict(((name, value) for name, value in six.iteritems(self._data.last_nag_times) if name in activated_ids))