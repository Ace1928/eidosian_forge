from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _ValidateServicesAreSubset(filtered_versions, all_versions):
    """Validate that each version in filtered_versions is also in all_versions.

  Args:
    filtered_versions: list of Version representing a filtered subset of
      all_versions.
    all_versions: list of Version representing all versions in the current
      project.

  Raises:
    VersionValidationError: If a service or version is not found.
  """
    for version in filtered_versions:
        if version.service not in [v.service for v in all_versions]:
            raise VersionValidationError('Service [{0}] not found.'.format(version.service))
        if version not in all_versions:
            raise VersionValidationError('Version [{0}/{1}] not found.'.format(version.service, version.id))