from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetInstancePresentationSpec(group_help):
    return presentation_specs.ResourcePresentationSpec('instance', GetInstanceResourceSpec(), group_help, required=True)