from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _ParseFromPluralLeaf(self, attribute_to_args_map, base_fallthroughs_map, anchor, parsed_args):
    """Helper for parsing a list of results using a single anchor value."""
    parsed_resources = []
    map_list = self._BuildFullFallthroughsMapList(anchor, attribute_to_args_map, base_fallthroughs_map, parsed_args)
    for fallthroughs_map in map_list:
        resource = self.Initialize(fallthroughs_map, parsed_args=parsed_args)
        if resource.result is not None:
            parsed_resources.append(resource)
    return parsed_resources