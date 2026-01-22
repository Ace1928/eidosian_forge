from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def CreateFindingArg():
    """Create finding as positional resource."""
    finding_spec_data = {'name': 'finding', 'collection': 'securitycenter.organizations.sources.findings', 'attributes': [{'parameter_name': 'organizationsId', 'attribute_name': 'organization', 'help': "(Optional) If the full resource name isn't provided e.g. organizations/123, then provide the\n              organization id which is the suffix of the organization. Example: organizations/123, the id is\n              123.", 'fallthroughs': [{'hook': 'googlecloudsdk.command_lib.scc.findings.flags:GetDefaultOrganization', 'hint': 'Set the organization property in configuration using `gcloud config set scc/organization`\n                  if it is not specified in command line.'}]}, {'parameter_name': 'sourcesId', 'attribute_name': 'source', 'help': "(Optional) If the full resource name isn't provided e.g. organizations/123/sources/456, then\n              provide the source id which is the suffix of the source.\n              Example: organizations/123/sources/456, the id is 456."}, {'parameter_name': 'findingId', 'attribute_name': 'finding', 'help': "Optional) If the full resource name isn't provided e.g.\n              organizations/123/sources/456/findings/789, then provide the finding id which is the suffix of\n              the finding. Example: organizations/123/sources/456/findings/789, the id is 789."}], 'disable_auto_completers': 'false'}
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to be used for the SCC (Security Command Center) command', name='finding', required=True, prefixes=False, positional=True, resource_data=finding_spec_data)]
    return concept_parsers.ConceptParser(arg_specs, [])