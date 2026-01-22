from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddHeterogeneousMigrationJobResourceArgs(parser, verb, required=False):
    """Add resource arguments for creating/updating a heterogeneous database mj.

  Args:
    parser: argparse.ArgumentParser, the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    required: boolean, whether source/dest resource args are required.
  """
    resource_specs = [presentation_specs.ResourcePresentationSpec('migration_job', GetMigrationJobResourceSpec(), 'The migration job {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--source', GetConnectionProfileResourceSpec(), 'ID of the source connection profile, representing the source database.', required=required, flag_name_overrides={'region': ''}), presentation_specs.ResourcePresentationSpec('--destination', GetConnectionProfileResourceSpec(), 'ID of the destination connection profile, representing the destination database.', required=required, flag_name_overrides={'region': ''}), presentation_specs.ResourcePresentationSpec('--conversion-workspace', GetConversionWorkspaceResourceSpec(), 'Name of the conversion workspaces to be used for the migration job', flag_name_overrides={'region': ''}), presentation_specs.ResourcePresentationSpec('--cmek-key', GetCmekKeyResourceSpec(), 'Name of the CMEK (customer-managed encryption key) used for the migration job', flag_name_overrides={'region': ''})]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--source.region': ['--region'], '--destination.region': ['--region'], '--conversion-workspace.region': ['--region'], '--cmek-key.region': ['--region']}).AddToParser(parser)