from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetDefaultProject():
    """Create a sourcerepo.projects resource of the default project."""
    return resources.REGISTRY.Create('sourcerepo.projects', projectsId=properties.VALUES.core.project.GetOrFail())