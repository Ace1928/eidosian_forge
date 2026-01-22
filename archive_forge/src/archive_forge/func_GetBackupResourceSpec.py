from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetBackupResourceSpec(resource_name='backup'):
    return concepts.ResourceSpec('gkebackup.projects.locations.backupPlans.backups', resource_name=resource_name, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LOCATION_RESOURCE_PARAMETER_ATTRIBUTE, backupPlansId=concepts.ResourceParameterAttributeConfig(name='backup-plan', fallthroughs=[deps.PropertyFallthrough(properties.VALUES.gkebackup.Property('backup_plan'))], help_text='Backup Plan name.'))