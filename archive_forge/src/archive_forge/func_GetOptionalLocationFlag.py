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
def GetOptionalLocationFlag():
    return concept_parsers.ConceptParser.ForResource('--location', GetLocationResourceSpec(), 'The Artifact Registry repository location. You can also set --location=all to list repositories across all locations. If you omit this flag, the default location is used if you set the artifacts/location property. Otherwise, omitting this flag lists repositories across all locations.', required=False)