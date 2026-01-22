from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def UserAccessibleProjectIDSet():
    """Retrieve the project IDs of projects the user can access.

  Returns:
    set of project IDs.
  """
    return set((p.projectId for p in projects_api.List()))