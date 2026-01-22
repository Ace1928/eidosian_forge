from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import enum
from googlecloudsdk.api_lib.app import exceptions as app_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def _LogNewWarnings(self, operation):
    if self.operation_metadata_type:
        new_warnings = GetWarningsFromOperation(operation, self.operation_metadata_type) - self.warnings_seen
        for warning in new_warnings:
            log.warning(warning + '\n')
            self.warnings_seen.add(warning)