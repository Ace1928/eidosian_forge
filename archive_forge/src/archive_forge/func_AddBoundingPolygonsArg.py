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
def AddBoundingPolygonsArg():
    return [base.Argument('--bounding-polygon', type=arg_parsers.ArgDict(spec={'vertices': list, 'normalized-vertices': list}, min_length=1), action='append', help='      Bounding polygon around the areas of interest in the reference image.\n      If this field is empty, the system will try to detect regions of interest.\n      This flag is repeatable to specify multiple bounding polygons. At most 10\n      bounding polygons will be used.\n\n      A bounding polygon can be specified by a list of vertices or normalized\n      vertices or both. A vertex (x, y) represents a 2D point in the image. x, y\n      are integers and are in the same scale as the original image.\n      The normalized vertex coordinates are relative to original image and\n      range from 0 to 1.\n\n      Because of the complexity of this flag, it should be specified\n      with the `--flags-file`. See $ gcloud topic flags-file for details.\n      See the examples section for how to use `--bounding-polygon` in\n      `--flags-file`.')]