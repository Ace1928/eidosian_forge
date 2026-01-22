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
def ListPendingApprovals(target_ref, pipeline_ref, output):
    """Lists the rollouts in pending approval state for the specified target.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_ref: protorpc.messages.Message, pipeline object.
    output: dictionary object

  Returns:
    The modified output object with the rollouts from the given pipeline pending
    approval on the given target.

  """
    try:
        pending_approvals = rollout_util.ListPendingRollouts(target_ref, pipeline_ref)
        pending_approvals_names = []
        for ro in pending_approvals:
            pending_approvals_names.append(ro.name)
        if pending_approvals_names:
            output['Pending Approvals'] = pending_approvals_names
    except apitools_exceptions.HttpError as error:
        log.debug('Failed to list pending approvals: ' + error.content)
    return output