from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddServiceResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Service Directory service."""
    name = 'service' if positional else '--service'
    return concept_parsers.ConceptParser.ForResource(name, GetServiceResourceSpec(), 'The Service Directory service {}'.format(verb), required=True).AddToParser(parser)