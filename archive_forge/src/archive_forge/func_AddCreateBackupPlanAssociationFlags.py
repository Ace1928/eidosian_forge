from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCreateBackupPlanAssociationFlags(parser):
    """Adds flags required to create a backup plan association."""
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('BACKUP_PLAN_ASSOCIATION', GetBackupPlanAssociationResourceSpec(), 'Name of the backup plan association to be created', required=True), presentation_specs.ResourcePresentationSpec('--backup-plan', GetBackupPlanResourceSpec(), 'Name of the backup plan to be applied to the specified resource.', flag_name_overrides={'location': '', 'project': ''}, required=True)], command_level_fallthroughs={'--backup-plan.location': ['BACKUP_PLAN_ASSOCIATION.location']}).AddToParser(parser)
    parser.add_argument('--resource', required=True, type=str, help='Resource to which the Backup Plan will be applied.')