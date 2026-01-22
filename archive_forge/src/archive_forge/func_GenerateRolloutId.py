from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GenerateRolloutId(to_target, release_ref):
    filter_str = ROLLOUT_IN_TARGET_FILTER_TEMPLATE.format(to_target)
    try:
        rollouts = rollout.RolloutClient().List(release_ref.RelativeName(), filter_str)
        return ComputeRolloutID(release_ref.Name(), to_target, rollouts)
    except apitools_exceptions.HttpError:
        raise cd_exceptions.ListRolloutsError(release_ref.RelativeName())