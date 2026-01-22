from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def ServiceAttachmentAttributeConfig(name='service_attachment'):
    return concepts.ResourceParameterAttributeConfig(name=name, help_text='The service attachment of the {resource}.')