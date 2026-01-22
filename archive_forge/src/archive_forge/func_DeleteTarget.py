from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def DeleteTarget(name):
    """Deletes a target resource by calling the delete target API.

  Args:
    name: str, target name.

  Returns:
    The operation message.
  """
    return target.TargetsClient().Delete(name)