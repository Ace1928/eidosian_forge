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
def _ParseFromPluralValue(self, attribute_to_args_map, base_fallthroughs_map, parsed_args, allow_empty=False):
    """Helper for parsing a list of resources from user input."""
    map_list = self._BuildFullFallthroughsMapList(attribute_to_args_map, base_fallthroughs_map, parsed_args=parsed_args)
    parsed_resources = []
    for fallthroughs_map in map_list:
        resource = self.Initialize(fallthroughs_map, parsed_args=parsed_args)
        parsed_resources.append(resource)
    if parsed_resources:
        return parsed_resources
    elif allow_empty:
        return []
    else:
        return self.Initialize(base_fallthroughs_map, parsed_args=parsed_args)