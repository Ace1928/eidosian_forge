from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudProviderValueValuesEnum(_messages.Enum):
    """Indicates which cloud provider was used in this simulation.

    Values:
      CLOUD_PROVIDER_UNSPECIFIED: The cloud provider is unspecified.
      GOOGLE_CLOUD_PLATFORM: The cloud provider is Google Cloud Platform.
      AMAZON_WEB_SERVICES: The cloud provider is Amazon Web Services.
      MICROSOFT_AZURE: The cloud provider is Microsoft Azure.
    """
    CLOUD_PROVIDER_UNSPECIFIED = 0
    GOOGLE_CLOUD_PLATFORM = 1
    AMAZON_WEB_SERVICES = 2
    MICROSOFT_AZURE = 3