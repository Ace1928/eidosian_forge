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
def IsVersionTriggererCompatible(image_version):
    """Checks if given `version` is compatible with triggerer .

  Args:
    image_version: image version str that includes airflow version.

  Returns:
    True if given airflow version is compatible with Triggerer(>=2.2.x)
    and Composer version is >=2.0.31 otherwise False
  """
    if image_version:
        version_item = _ImageVersionItem(image_version)
        if IsVersionComposer3Compatible(image_version):
            return True
        if version_item and version_item.airflow_ver and version_item.composer_ver:
            airflow_version = version_item.airflow_ver
            composer_version = version_item.composer_ver
            if composer_version == 'latest':
                composer_version = COMPOSER_LATEST_VERSION_PLACEHOLDER
            return IsVersionInRange(composer_version, flags.MIN_TRIGGERER_COMPOSER_VERSION, None, True) and IsVersionInRange(airflow_version, flags.MIN_TRIGGERER_AIRFLOW_VERSION, None, True)
    return False