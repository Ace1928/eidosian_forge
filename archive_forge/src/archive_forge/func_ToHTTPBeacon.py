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
def ToHTTPBeacon(self, timer):
    """Collect the required clearcut HTTP beacon."""
    clearcut_request = dict(self.request_params)
    clearcut_request['request_time_ms'] = GetTimeMillis()
    event_latency, sub_event_latencies = self.Timings(timer)
    command_latency_set = False
    for concord_event, _ in self._clearcut_concord_timed_events:
        if concord_event['event_type'] is _COMMANDS_CATEGORY and command_latency_set:
            continue
        concord_event['latency_ms'] = event_latency
        concord_event['sub_event_latency_ms'] = sub_event_latencies
        command_latency_set = concord_event['event_type'] is _COMMANDS_CATEGORY
    clearcut_request['log_event'] = []
    for concord_event, event_time_ms in self._clearcut_concord_timed_events:
        clearcut_request['log_event'].append({'source_extension_json': json.dumps(concord_event, sort_keys=True), 'event_time_ms': event_time_ms})
    data = json.dumps(clearcut_request, sort_keys=True)
    headers = {'user-agent': self._user_agent}
    return (_CLEARCUT_ENDPOINT, 'POST', data, headers)