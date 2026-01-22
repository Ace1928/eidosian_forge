from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.transfer.appliances import regions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _get_filter_clause_from_resources(filter_key, resource_refs):
    if not resource_refs:
        return ''
    filter_list = ['{}:{}'.format(filter_key, ref.RelativeName()) for ref in resource_refs]
    resource_list = ' OR '.join(filter_list)
    return '({})'.format(resource_list)