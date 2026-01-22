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
def GetRepoArgFromBeta():
    return concept_parsers.ConceptParser.ForResource('repository', GetBetaRepoResourceSpec(), 'The Artifact Registry repository. If not specified, the current artifacts/repository is used.', required=True)