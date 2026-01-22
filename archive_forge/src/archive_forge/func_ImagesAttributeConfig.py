from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def ImagesAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='image', help_text='Name of the image.')