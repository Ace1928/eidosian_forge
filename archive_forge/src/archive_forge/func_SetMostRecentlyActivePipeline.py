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
def SetMostRecentlyActivePipeline(pipeline_ref, rollout):
    """Retrieves latest rollout and release information for a delivery pipeline.

  Args:
    pipeline_ref: protorpc.messages.Message a DeliveryPipeline object.
    rollout: protorpc.messages.Message a Rollout object.

  Returns:
    A content directory.

  """
    output = [{pipeline_ref.RelativeName(): SetCurrentReleaseAndRollout(rollout, {})}]
    return output