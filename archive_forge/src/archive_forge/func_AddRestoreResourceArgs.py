from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddRestoreResourceArgs(parser):
    """Add backup resource args (source, destination) for restore command."""
    arg_specs = [presentation_specs.ResourcePresentationSpec('--source', GetBackupResourceSpec(), 'TEXT', required=True, flag_name_overrides={'instance': '--source-instance', 'backup': '--source-backup'}), presentation_specs.ResourcePresentationSpec('--destination', GetDatabaseResourceSpec(), 'TEXT', required=True, flag_name_overrides={'instance': '--destination-instance', 'database': '--destination-database'})]
    concept_parsers.ConceptParser(arg_specs).AddToParser(parser)