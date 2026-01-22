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
def ImageVersionFromAirflowVersion(new_airflow_version, cur_image_version=None):
    """Converts airflow-version string into a image-version string."""
    is_composer3 = cur_image_version and IsVersionComposer3Compatible(cur_image_version)
    composer_ver = _ImageVersionItem(cur_image_version).composer_ver if is_composer3 else 'latest'
    return _ImageVersionItem(composer_ver=composer_ver, airflow_ver=new_airflow_version).GetImageVersionString()