from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetOnDemandScanningLocationFlag():
    return base.Argument('--location', choices={'us': 'Perform analysis in the US', 'europe': 'Perform analysis in Europe', 'asia': 'Perform analysis in Asia'}, default='us', help='The API location in which to perform package analysis. Consider choosing a location closest to where you are located. Proximity to the container image does not affect response time.')