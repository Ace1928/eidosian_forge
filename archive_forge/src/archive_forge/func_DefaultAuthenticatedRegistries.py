from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core.docker import client_lib as client_utils
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import semver
import six
def DefaultAuthenticatedRegistries(include_artifact_registry=False):
    """Return list of default gcloud credential helper registires."""
    if include_artifact_registry:
        return constants.DEFAULT_REGISTRIES_TO_AUTHENTICATE + constants.REGIONAL_AR_REGISTRIES
    else:
        return constants.DEFAULT_REGISTRIES_TO_AUTHENTICATE