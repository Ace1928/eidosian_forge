from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBillingAccountResourceArg(parser, description):
    concept_parsers.ConceptParser.ForResource('--billing-account', GetBillingAccountResourceSpec(), description, required=True).AddToParser(parser)