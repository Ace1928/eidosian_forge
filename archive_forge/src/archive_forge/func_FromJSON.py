from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import component_status
import six
@classmethod
def FromJSON(cls, name, json_map):
    """Instantiate a new ModuleStatus from JSON.

    Args:
      name: the name of the Module
      json_map: a JSON dict mapping component name to the JSON representation of
        ComponentStatus (see ComponentStatus.FromJSON)

    Returns:
      a ModuleStatus object
    """
    comps = sorted([component_status.ComponentStatus.FromJSON(comp_name, json) for comp_name, json in json_map['components'].items()], key=lambda c: c.name)
    return cls(name=name, components=comps)