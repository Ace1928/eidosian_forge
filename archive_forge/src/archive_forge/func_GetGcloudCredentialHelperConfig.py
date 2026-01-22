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
def GetGcloudCredentialHelperConfig(registries=None, include_artifact_registry=False):
    """Gets the credHelpers Docker config entry for gcloud supported registries.

  Returns a Docker configuration JSON entry that will register gcloud as the
  credential helper for all Google supported Docker registries.

  Args:
      registries: list, the registries to create the mappings for. If not
        supplied, will use DefaultAuthenticatedRegistries().
      include_artifact_registry: bool, whether to include all Artifact Registry
        domains as well as GCR domains registries when called with no list of
        registries to add.

  Returns:
    The config used to register gcloud as the credential helper for all
    supported Docker registries.
  """
    registered_helpers = BuildOrderedCredentialHelperRegistries(registries or DefaultAuthenticatedRegistries(include_artifact_registry))
    return {CREDENTIAL_HELPER_KEY: registered_helpers}