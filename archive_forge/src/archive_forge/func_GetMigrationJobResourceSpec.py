from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetMigrationJobResourceSpec(resource_name='migration_job'):
    return concepts.ResourceSpec('datamigration.projects.locations.migrationJobs', resource_name=resource_name, migrationJobsId=MigrationJobAttributeConfig(name=resource_name), locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)