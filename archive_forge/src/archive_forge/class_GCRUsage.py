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
@dataclasses.dataclass(frozen=True)
class GCRUsage:
    """GCRUsage represents usage for a GCR repo.

  Attributes:
    repository: A GCR repo name.
    usage: Usage for the repo.
  """
    repository: str
    usage: str