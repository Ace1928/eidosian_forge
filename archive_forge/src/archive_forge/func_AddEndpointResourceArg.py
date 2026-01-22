from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEndpointResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Service Directory endpoint."""
    name = 'endpoint' if positional else '--endpoint'
    return concept_parsers.ConceptParser.ForResource(name, GetEndpointResourceSpec(), 'The Service Directory endpoint {}'.format(verb), required=True).AddToParser(parser)