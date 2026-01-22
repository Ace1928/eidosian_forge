from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddParentEntryResourceArg(parser):
    """Adds a resource argument for a Dataplex Entry parent."""
    entry_data = yaml_data.ResourceYAMLData.FromPath('dataplex.entry')
    return concept_parsers.ConceptParser.ForResource('--parent-entry', concepts.ResourceSpec.FromYaml(entry_data.GetData()), 'Arguments and flags that define the parent Entry you want to reference.', command_level_fallthroughs={'location': ['--location'], 'entry_group': ['--entry_group']}, flag_name_overrides={'location': '', 'entry_group': ''}).AddToParser(parser)