from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def TargetReferenceFromName(target_name):
    """Creates a target reference from full name.

  Args:
    target_name: str, target resource name.

  Returns:
    Target reference.
  """
    return resources.REGISTRY.ParseRelativeName(target_name, collection=_SHARED_TARGET_COLLECTION)