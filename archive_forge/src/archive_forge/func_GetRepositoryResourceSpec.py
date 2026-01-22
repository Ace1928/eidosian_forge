from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetRepositoryResourceSpec():
    return concepts.ResourceSpec('securesourcemanager.projects.locations.repositories', resource_name='repository', repositoriesId=RepositoryAttributeConfig(), locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)