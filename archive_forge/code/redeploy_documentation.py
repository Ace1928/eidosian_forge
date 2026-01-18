from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import promote_util
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
Gets the current release in the target.

  Args:
    pipeline_ref: pipeline_ref: protorpc.messages.Message, pipeline object.
    target_ref: target_ref: protorpc.messages.Message, target object.
    filter_str: Filter string to use when listing rollouts.

  Returns:
    The most recent release with the given pipeline and target with a rollout
    that is redeployable.

  Raises:
    core.Error: Target has no rollouts or rollouts in target are not
                redeployable.
  