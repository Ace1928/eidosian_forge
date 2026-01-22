from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def RepoAttributeConfig():
    """Get Cloud Source Repo resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='repo', help_text='Name of the repository.')