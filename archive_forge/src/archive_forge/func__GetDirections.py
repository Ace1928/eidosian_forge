from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _GetDirections(name, full_fallthroughs_map, concept_specs):
    """Aggregates directions on how to specify each type of resource."""
    directions = []
    for spec in concept_specs:
        attribute_directions = _GetAttributeDirections(spec.attributes, full_fallthroughs_map)
        directions.append(f'\nTo specify [{name}] as type {spec.collection}, specify only the following attributes.')
        directions.append(attribute_directions)
    return '\n\n'.join(directions)