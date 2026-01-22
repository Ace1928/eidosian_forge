from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceSnapshot(_messages.Message):
    """Result containing the properties and count of a ComplianceSnapshot
  request.

  Enums:
    CloudProviderValueValuesEnum: The cloud provider for the compliance
      snapshot.

  Fields:
    category: The category of Findings matching.
    cloudProvider: The cloud provider for the compliance snapshot.
    complianceStandard: The compliance standard (ie CIS).
    complianceVersion: The compliance version (ie 1.3) in CIS 1.3.
    count: Total count of findings for the given properties.
    leafContainerResource: The leaf container resource name that is closest to
      the snapshot.
    name: The compliance snapshot name. Format:
      //sources//complianceSnapshots/
    projectDisplayName: The CRM resource display name that is closest to the
      snapshot the Findings belong to.
    snapshotTime: The snapshot time of the snapshot.
  """

    class CloudProviderValueValuesEnum(_messages.Enum):
        """The cloud provider for the compliance snapshot.

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
    category = _messages.StringField(1)
    cloudProvider = _messages.EnumField('CloudProviderValueValuesEnum', 2)
    complianceStandard = _messages.StringField(3)
    complianceVersion = _messages.StringField(4)
    count = _messages.IntegerField(5)
    leafContainerResource = _messages.StringField(6)
    name = _messages.StringField(7)
    projectDisplayName = _messages.StringField(8)
    snapshotTime = _messages.StringField(9)