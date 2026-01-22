from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def _GetAttestorResourceSpec():
    return concepts.ResourceSpec('binaryauthorization.projects.attestors', resource_name='attestor', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, attestorsId=concepts.ResourceParameterAttributeConfig(name='name', help_text='The ID of the {resource}.'))