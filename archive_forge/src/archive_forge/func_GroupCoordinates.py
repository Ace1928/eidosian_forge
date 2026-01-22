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
def GroupCoordinates(coordinates):
    if len(coordinates) % 2 != 0:
        raise BoundingPolygonFormatError('There must be an even number of values in the list.')
    grouped_coordinates = []
    for i in range(0, len(coordinates), 2):
        grouped_coordinates.append((coordinates[i], coordinates[i + 1]))
    return grouped_coordinates