from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import image_versions_util as image_version_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core.util import semver
def IsVersionAirflowCommandsApiCompatible(image_version):
    """Checks if given `version` is compatible with Composer Airflow Commands API.

  Args:
    image_version: image version str that includes Composer version.

  Returns:
    True if Composer version is compatible with Aiflow Commands API,
    otherwise False.
  """
    if image_version:
        version_item = _ImageVersionItem(image_version)
        if version_item and version_item.composer_ver:
            composer_version = version_item.composer_ver
            return IsVersionInRange(composer_version, flags.MIN_COMPOSER_RUN_AIRFLOW_CLI_VERSION, None, True)
    return False