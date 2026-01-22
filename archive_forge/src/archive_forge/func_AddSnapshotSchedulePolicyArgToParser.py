from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSnapshotSchedulePolicyArgToParser(parser, positional=False, required=True, name=None, group=None):
    """Sets up an argument for the snapshot schedule policy resource."""
    if not name:
        if positional:
            name = 'snapshot_schedule_policy'
        else:
            name = '--snapshot-schedule-policy'
    policy_data = yaml_data.ResourceYAMLData.FromPath('bms.snapshot_schedule_policy')
    resource_spec = concepts.ResourceSpec.FromYaml(policy_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, group=group, concept_spec=resource_spec, required=required, flag_name_overrides={'region': ''}, group_help='snapshot_schedule_policy.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)