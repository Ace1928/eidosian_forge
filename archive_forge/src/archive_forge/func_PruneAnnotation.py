from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def PruneAnnotation(resource):
    """Garbage-collect items in the run.googleapis.com/secrets annotation.

  Args:
    resource: k8s_object resource to be modified.
  """
    in_use = _InUse(resource)
    formatted_annotation = _GetSecretsAnnotation(resource)
    to_keep = {alias: rs for alias, rs in ParseAnnotation(formatted_annotation).items() if alias in in_use}
    _SetSecretsAnnotation(resource, _FormatAnnotation(to_keep))