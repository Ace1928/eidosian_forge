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
def _BuildUpgradeCandidateList(location_ref, image_version_id, python_version, release_track=base.ReleaseTrack.GA):
    """Builds a list of eligible image version upgrades."""
    image_version_service = image_version_api_util.ImageVersionService(release_track)
    image_version_item = _ImageVersionItem(image_version_id)
    available_upgrades = []
    if IsVersionComposer3Compatible(image_version_id) or CompareVersions(MIN_UPGRADEABLE_COMPOSER_VER, image_version_item.composer_ver) <= 0:
        for version in image_version_service.List(location_ref):
            if _ValidateCandidateImageVersionId(image_version_id, version.imageVersionId).upgrade_valid and (not python_version or python_version in version.supportedPythonVersions):
                available_upgrades.append(version)
    else:
        raise InvalidImageVersionError('This environment does not support upgrades.')
    return available_upgrades