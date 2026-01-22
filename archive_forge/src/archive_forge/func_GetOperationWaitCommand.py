from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.core.console import progress_tracker as console_progress_tracker
from googlecloudsdk.core.util import retry
@staticmethod
def GetOperationWaitCommand(operation_ref):
    return 'gcloud beta sql operations wait --project {0} {1}'.format(operation_ref.project, operation_ref.operation)