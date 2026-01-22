from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def FreeTrialAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='free-trial', help_text='Procurement free trial for the {resource}.')