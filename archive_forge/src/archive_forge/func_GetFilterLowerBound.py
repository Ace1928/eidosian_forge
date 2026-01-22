from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def GetFilterLowerBound(self):
    """The log message filter which keeps out messages which are too old.

    Returns:
        The lower bound filter text that we should use.
    """
    if self.need_insert_id_in_lb_filter:
        return '((timestamp="{0}" AND insertId>"{1}") OR timestamp>"{2}")'.format(self.timestamp, self.insert_id, self.timestamp)
    else:
        return 'timestamp>="{0}"'.format(self.timestamp)