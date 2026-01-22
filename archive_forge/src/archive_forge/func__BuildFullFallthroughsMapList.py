from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _BuildFullFallthroughsMapList(self, attribute_to_args_map, base_fallthroughs_map, parsed_args=None):
    """Builds fallthrough map for each anchor value specified in a list.

    For each anchor value, create a falthrough map to derive the rest
    of the resource params. For each attribute, adds flag fallthroughs
    and fully specified anchor fallthroughs. For each attribute,
    adds default flag fallthroughs and fully specified anchor fallthroughs.

    Args:
      attribute_to_args_map: {str: str}, A map of attribute names to the names
        of their associated flags.
      base_fallthroughs_map: FallthroughsMap, A map of attribute names to
        non-argument fallthroughs, including command-level fallthroughs.
      parsed_args: Namespace, used to parse the anchor value and derive
        fully specified fallthroughs.

    Returns:
      list[FallthroughsMap], fallthrough map for each anchor value
    """
    fallthroughs_map = {**base_fallthroughs_map}
    deps_map_util.AddFlagFallthroughs(fallthroughs_map, self.attributes, attribute_to_args_map)
    deps_map_util.PluralizeFallthroughs(fallthroughs_map, self.anchor.name)
    map_list = deps_map_util.CreateValueFallthroughMapList(fallthroughs_map, self.anchor.name, parsed_args)
    for full_map in map_list:
        deps_map_util.AddAnchorFallthroughs(full_map, self.attributes, self.anchor, self.collection_info, full_map.get(self.anchor.name, []))
    return map_list