from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
import six
def CheckRuntimeVersion(model=None, version=None):
    """Check if runtime-version is more than 1.8."""
    framework, runtime_version = GetRuntimeVersion(model, version)
    if framework == 'TENSORFLOW':
        release, version = map(int, runtime_version.split('.'))
        return release == 1 and version >= 8 or release > 1
    else:
        return False