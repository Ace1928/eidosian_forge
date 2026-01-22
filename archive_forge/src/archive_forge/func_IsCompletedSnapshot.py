from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def IsCompletedSnapshot(bp):
    return (not bp.action or bp.action == self.BreakpointAction(self.SNAPSHOT_TYPE)) and bp.isFinalState and (not (bp.status and bp.status.isError))