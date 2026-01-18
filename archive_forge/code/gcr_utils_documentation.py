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
Converts a project ID to a GCR bucket suffix.

  Args:
    project_id: The project ID.

  Returns:
    A string representing the suffix of GCR buckets in the project. The suffix
    format is different for normal projects and domain-scoped projects. For
    example:

    my-proj           -> artifacts.my-proj.appspot.com
    my-domain:my-proj -> artifacts.my-proj.my-domain.a.appspot.com
  