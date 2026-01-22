import dataclasses
from typing import Iterator
from apitools.base.py import list_pager
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib.asset import client_util as asset_client_util
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def CheckGCRUsage(repo: docker_name.Repository) -> GCRUsage:
    """Checks usage for a GCR repo.

  Args:
    repo: A docker repository.

  Returns:
    A GCRUsage object.
  """
    try:
        with docker_image.FromRegistry(basic_creds=util.CredentialProvider(), name=repo, transport=util.Http()) as r:
            return GCRUsage(str(repo), r.check_usage_only())
    except (docker_http.V2DiagnosticException, docker_http.TokenRefreshException) as e:
        return GCRUsage(str(repo), str(e))