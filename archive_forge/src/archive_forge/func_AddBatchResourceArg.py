from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddBatchResourceArg(parser, verb, api_version, use_location=False):
    """Adds batch resource argument to parser."""

    def BatchConfig():
        return concepts.ResourceParameterAttributeConfig(name='batch', help_text='Batch job ID.')

    def GetBatchResourceSpec(api_version):
        return concepts.ResourceSpec('dataproc.projects.locations.batches', api_version=api_version, resource_name='batch', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=_LocationAttributeConfig() if use_location else _RegionAttributeConfig(), batchesId=BatchConfig())
    concept_parsers.ConceptParser.ForResource('batch', GetBatchResourceSpec(api_version), 'ID of the batch job to {0}.'.format(verb), required=True).AddToParser(parser)