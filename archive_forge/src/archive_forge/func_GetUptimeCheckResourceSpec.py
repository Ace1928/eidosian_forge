from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetUptimeCheckResourceSpec():
    return concepts.ResourceSpec('monitoring.projects.uptimeCheckConfigs', resource_name='uptime check or synthetic monitor', uptimeCheckConfigsId=UptimeCheckAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)