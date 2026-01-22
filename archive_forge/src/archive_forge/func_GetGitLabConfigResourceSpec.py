from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.core import properties
def GetGitLabConfigResourceSpec():
    return concepts.ResourceSpec('cloudbuild.projects.locations.gitLabConfigs', api_version='v1', resource_name='gitLabConfig', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=RegionAttributeConfig(), gitLabConfigsId=GitLabConfigAttributeConfig())