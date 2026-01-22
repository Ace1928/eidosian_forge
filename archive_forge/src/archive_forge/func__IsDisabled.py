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
@staticmethod
def _IsDisabled():
    """Returns whether metrics collection should be disabled."""
    if _MetricsCollector._disabled_cache is None:
        if '_ARGCOMPLETE' in os.environ:
            _MetricsCollector._disabled_cache = True
        elif not properties.IsDefaultUniverse():
            _MetricsCollector._disabled_cache = True
        else:
            disabled = properties.VALUES.core.disable_usage_reporting.GetBool()
            if disabled is None:
                disabled = config.INSTALLATION_CONFIG.disable_usage_reporting
            _MetricsCollector._disabled_cache = disabled
    return _MetricsCollector._disabled_cache