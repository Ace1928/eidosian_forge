from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def VerifyScopeForSearch(scope):
    """Perform permissive validation of the search scope.

  This validation is required although the API server contains similar request
  validation.
  The reason is that a malformed scope will be translated into an
  invalid URL, resulting in 404. For example, scope "projects/123/abc/" is
  translated to
  "https://cloudasset.googleapis.com/v1p1beta1/projects/123/abc/resources:searchAll".(404)
  However our OnePlatform API only accepts URL in format:
  "https://cloudasset.googleapis.com/v1p1beta1/*/*/resources:searchAll"

  Args:
    scope: the scope string of a search request.
  """
    if not re.match('^[^/#?]+/[^/#?]+$', scope):
        raise gcloud_exceptions.InvalidArgumentException('--scope', 'A valid scope should be: projects/{PROJECT_ID}, projects/{PROJECT_NUMBER}, folders/{FOLDER_NUMBER} or organizations/{ORGANIZATION_NUMBER}.')