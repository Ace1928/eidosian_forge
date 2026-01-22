from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def CreateRepo(self, repo_resource):
    """Creates a repo.

    Args:
      repo_resource: (Resource) A resource representing the repo to create.
    Returns:
      (messages.Repo) The full definition of the new repo, as reported by
        the server.
    """
    parent = resources.REGISTRY.Create('sourcerepo.projects', projectsId=repo_resource.projectsId)
    request = self.messages.SourcerepoProjectsReposCreateRequest(parent=parent.RelativeName(), repo=self.messages.Repo(name=repo_resource.RelativeName()))
    return self._client.projects_repos.Create(request)