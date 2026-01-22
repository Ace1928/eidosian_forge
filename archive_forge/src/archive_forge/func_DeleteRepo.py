from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def DeleteRepo(self, repo_resource):
    """Deletes a repo.

    Args:
      repo_resource: (Resource) A resource representing the repo to create.
    """
    request = self.messages.SourcerepoProjectsReposDeleteRequest(name=repo_resource.RelativeName())
    self._client.projects_repos.Delete(request)