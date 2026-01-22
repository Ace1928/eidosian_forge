from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def _RegisterReqestForLogging(self, path, response_code, start_time_usec, request_size_bytes):
    """Registers a request for client deploy logging purposes."""
    end_time_usec = self.GetCurrentTimeUsec()
    self.requests.append(client_deployinfo.Request(path=path, response_code=response_code, start_time_usec=start_time_usec, end_time_usec=end_time_usec, request_size_bytes=request_size_bytes))