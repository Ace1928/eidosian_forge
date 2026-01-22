from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
Initializes an AppengineApiClient using the specified API version.

    Uses the api_client_overrides/appengine property to determine which client
    version to use if api_version is not set. Additionally uses the
    api_endpoint_overrides/appengine property to determine the server endpoint
    for the App Engine API.

    Args:
      api_version: The api version override.

    Returns:
      An AppengineApiClient used by gcloud to communicate with the App Engine
      API.

    Raises:
      ValueError: If default_version does not correspond to a supported version
      of the API.
    