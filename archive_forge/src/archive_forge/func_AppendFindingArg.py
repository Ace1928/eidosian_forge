from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AppendFindingArg():
    """Add finding as positional resource."""
    finding_spec_data = yaml_data.ResourceYAMLData.FromPath('scc.finding')
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to be used for the SCC (Security Command Center) command', name='finding', required=True, prefixes=False, positional=True, resource_data=finding_spec_data.GetData())]
    return [concept_parsers.ConceptParser(arg_specs, [])]