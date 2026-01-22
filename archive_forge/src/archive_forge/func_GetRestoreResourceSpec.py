from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetRestoreResourceSpec(resource_name='restore'):
    return concepts.ResourceSpec('gkebackup.projects.locations.restorePlans.restores', api_version='v1', resource_name=resource_name, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LOCATION_RESOURCE_PARAMETER_ATTRIBUTE, restorePlansId=concepts.ResourceParameterAttributeConfig(name='restore-plan', fallthroughs=[deps.PropertyFallthrough(properties.VALUES.gkebackup.Property('restore_plan'))], help_text='Restore Plan name.'))