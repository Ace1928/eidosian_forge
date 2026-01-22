from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def _ValidateAndExtractFromBoundingPolygonArgs(bounding_polygon_arg):
    """Extracts coordinates from users' input."""
    if not bounding_polygon_arg:
        return []
    coordinates = bounding_polygon_arg.split(',')
    grouped_coordinates = GroupCoordinates(coordinates)
    if _IsPolygonSpecifiedAsVertex(coordinates):
        return [Vertex(x, y) for x, y in grouped_coordinates]
    if _IsPolygonSpecifiedAsNormalizedVertex(coordinates):
        return [NormalizedVertex(x, y) for x, y in grouped_coordinates]
    raise BoundingPolygonFormatError('Coordinates of normalized vertex should have decimal points, Coordinates of vertex should be integers and cannot have decimal points.')