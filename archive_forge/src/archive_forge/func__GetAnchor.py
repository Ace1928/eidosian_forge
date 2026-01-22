from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _GetAnchor(self):
    leaf_anchors = set((attr for attr in self.attributes if self.IsLeafAnchor(attr)))
    if len(leaf_anchors) != 1:
        anchor_names = ', '.join([attr.name for attr in leaf_anchors])
        raise ConfigurationError(f'Could not find single achor value for multitype resource. Resource {self.name} has multiple leaf anchors: [{anchor_names}].')
    return leaf_anchors.pop()