from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def SetPipelineReleaseRollout(target_ref, pipeline_ref):
    """Retrieves latest rollout and release information for a single delivery pipeline.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_ref: protorpc.messages.Message, DeliveryPipeline object

  Returns:
    A content directory.

  """
    current_rollout = target_util.GetCurrentRollout(target_ref, pipeline_ref)
    output = {}
    output = SetCurrentReleaseAndRollout(current_rollout, output)
    return output