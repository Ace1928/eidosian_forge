from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetRevisionNumber(revision_full_name):
    """Returns the revision number from the revision name.

     e.g. - returns 12 for
     projects/p1/locations/l1/deployments/d1/revisions/r-12.

  Args:
    revision_full_name: string, the fully qualified name of the revision, e.g.
      "projects/p/locations/l/deployments/d/revisions/r-12".

  Returns:
    a revision number.
  """
    revision_ref = resources.REGISTRY.Parse(revision_full_name, collection='config.projects.locations.deployments.revisions')
    revision_short_name = revision_ref.Name()
    return int(revision_short_name[2:])