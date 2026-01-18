from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
List target resources by calling the list target API.

  Args:
    parent_name: str, the name of the collection that owns the targets.

  Returns:
    List of targets returns from target list API call.
  