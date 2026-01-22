from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetServiceAttachmentResourceSpec(resource_name='service_attachment'):
    return concepts.ResourceSpec('compute.serviceAttachments', resource_name=resource_name, serviceAttachment=ServiceAttachmentAttributeConfig(name=resource_name), project=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)