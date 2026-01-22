from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def UpdateAutoscalingScalingConfig(unused_ref, args, req):
    """Sets autoscalingEnabled to true if the service specified a min scaling factor, max scaling factor, or both.

  Args:
    args: The request arguments.
    req: A request with `service` field.

  Returns:
    A request with a modified scaling config.
  """
    if args.min_scaling_factor or args.max_scaling_factor:
        req.service.scalingConfig.autoscalingConfig.autoscalingEnabled = True
    return req