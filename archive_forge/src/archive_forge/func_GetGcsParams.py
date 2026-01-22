from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def GetGcsParams(arg_name, path):
    """Returns information for a Google Cloud Storage object.

  Args:
      arg_name: The name of the argument whose value may be a GCS object path.
      path: A string whose value may be a GCS object path.
  """
    obj_ref = None
    for prefix in _GCS_PREFIXES:
        if path.startswith(prefix):
            obj_ref = resources.REGISTRY.Parse(path)
            break
    if not obj_ref:
        return None
    if not hasattr(obj_ref, 'bucket') or not hasattr(obj_ref, 'object'):
        raise exceptions.InvalidArgumentException(arg_name, 'The provided Google Cloud Storage path [{}] is invalid.'.format(path))
    obj_str = obj_ref.object.split('#')
    if len(obj_str) != 2 or not obj_str[1].isdigit():
        raise exceptions.InvalidArgumentException(arg_name, 'The provided Google Cloud Storage path [{}] does not contain a valid generation number.'.format(path))
    return {'bucket': obj_ref.bucket, 'object': obj_str[0], 'generationNumber': int(obj_str[1])}