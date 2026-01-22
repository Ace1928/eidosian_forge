from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetBackupPlanResourceSpec():
    return concepts.ResourceSpec('backupdr.projects.locations.backupPlans', resource_name='Backup Plan', locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, backupPlansId=concepts.ResourceParameterAttributeConfig(name='backup_plan', help_text='some help text'), disable_auto_completers=False)