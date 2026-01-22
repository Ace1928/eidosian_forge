from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import module_status
import six
Instantiates a new ApplicationStatus from a JSON.

    Args:
      json_map: a JSON dict mapping module name to the JSON representation of
        ModuleStatus (see ModuleStatus.FromJSON)

    Returns:
      a new ApplicationStatus object
    