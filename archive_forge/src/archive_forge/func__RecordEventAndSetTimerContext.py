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
def _RecordEventAndSetTimerContext(category, action, label, value=0, flag_names=None, error=None, error_extra_info_json=None):
    """Common code for processing an event."""
    collector = _MetricsCollector.GetCollector()
    if not collector:
        return
    if _MetricsCollector.test_group and category is not _ERROR_CATEGORY:
        label = _MetricsCollector.test_group
    event = _Event(category=category, action=action, label=label, value=value)
    collector.Record(event, flag_names=flag_names, error=error, error_extra_info_json=error_extra_info_json)
    if category in [_COMMANDS_CATEGORY, _EXECUTIONS_CATEGORY]:
        collector.SetTimerContext(category, action, flag_names=flag_names)
    elif category in [_ERROR_CATEGORY, _HELP_CATEGORY, _TEST_EXECUTIONS_CATEGORY]:
        collector.SetTimerContext(category, action, label, flag_names=flag_names)