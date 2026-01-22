from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _MakeProjectName(self, project):
    project_ref = resources.REGISTRY.Create(API_NAME + '.projects', projectId=project)
    return project_ref.RelativeName()