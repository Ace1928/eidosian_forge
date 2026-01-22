from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.source import git
from googlecloudsdk.api_lib.source import sourcerepo
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store as c_store
def ActionIfMirror(self, project, repo, mirror_url):
    """Raises an exception if the repository is a mirror."""
    message = 'Repository "{repo}" in project "{prj}" is a mirror. Clone the mirrored repository directly with \n$ git clone {url}'.format(repo=repo, prj=project, url=mirror_url)
    raise c_exc.InvalidArgumentException('REPOSITORY_NAME', message)