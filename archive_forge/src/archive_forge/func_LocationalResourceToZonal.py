from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def LocationalResourceToZonal(path):
    """Converts a resource identifier (possibly a full URI) to the zonal format.

  e.g., container.projects.locations.clusters (like
  projects/foo/locations/us-moon1/clusters/my-cluster) ->
  container.projects.zones.clusters (like
  projects/foo/zones/us-moon1/clusters/my-cluster). While the locational format
  is newer, we have to use a single one because the formats have different
  fields. This allows either to be input, but the code will use entirely the
  zonal format.

  Args:
    path: A string resource name, possibly a URI (i.e., self link).

  Returns:
    The string identifier converted to zonal format if applicable. Unchanged if
    not applicable (i.e., not a full path or already in zonal format).
  """
    return path.replace('/locations/', '/zones/')