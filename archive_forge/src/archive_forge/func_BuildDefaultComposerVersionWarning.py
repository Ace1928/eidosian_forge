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
def BuildDefaultComposerVersionWarning(image_version, airflow_version):
    """Builds warning message about using default Composer version."""
    message = '{} resolves to Cloud Composer current default version, which is presently Composer 2 and is subject to further changes in the future. Consider using --image-version=composer-A-airflow-X[.Y[.Z]]. More info at https://cloud.google.com/composer/docs/concepts/versioning/composer-versioning-overview#version-aliases'
    if airflow_version:
        return message.format('Using --airflow-version=X[.Y[.Z]]')
    if image_version:
        return message.format('Using --image-version=composer-latest-airflow-X[.Y[.Z]]')
    return message.format('Not defining --image-version')