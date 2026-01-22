from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetPatchDeploymentResourceSpec():
    return concepts.ResourceSpec('osconfig.projects.patchDeployments', resource_name='patch_deployment', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, patchDeploymentsId=PatchDeploymentAttributeConfig())