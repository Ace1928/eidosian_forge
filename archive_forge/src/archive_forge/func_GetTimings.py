from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def GetTimings(self):
    """Returns the timings for the recorded events."""
    timings = []
    for event in self.__events:
        timings.append((event.name, event.time_millis - self.__start))
    timings.extend([(_LOCAL_EVENT, self.__total_local_duration), (_REMOTE_EVENT, self.__total_rpc_duration)])
    return timings