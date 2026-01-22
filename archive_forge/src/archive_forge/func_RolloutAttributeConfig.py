from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def RolloutAttributeConfig():
    """Creates the rollout resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='rollout', help_text='The rollout associated with the {resource}.')