from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddScanRunResourceArg(parser):
    concept_parsers.ConceptParser.ForResource('scan_run', GetScanRunResourceSpec(), 'The scan run resource which all the findings belong.', required=True).AddToParser(parser)