from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddManagementServerResourceArg(parser, help_text):
    """Adds an argument for management server to parser."""
    name = 'management_server'
    concept_parsers.ConceptParser.ForResource(name, GetManagementServerResourceSpec(), help_text, required=True).AddToParser(parser)