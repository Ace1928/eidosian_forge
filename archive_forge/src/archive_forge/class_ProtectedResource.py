from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ProtectedResource(_messages.Message):
    """Metadata about a resource protected by a Cloud KMS key.

  Messages:
    LabelsValue: A key-value pair of the resource's labels (v1) to their
      values.

  Fields:
    cloudProduct: The Cloud product that owns the resource. Example: `compute`
    createTime: Output only. The time at which this resource was created. The
      granularity is in seconds. Timestamp.nanos will always be 0.
    cryptoKeyVersion: The name of the Cloud KMS [CryptoKeyVersion](https://clo
      ud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cry
      ptoKeys.cryptoKeyVersions?hl=en) used to protect this resource via CMEK.
      This field is empty if the Google Cloud product owning the resource does
      not provide key version data to Asset Inventory. If there are multiple
      key versions protecting the resource, then this is same value as the
      first element of crypto_key_versions.
    cryptoKeyVersions: The names of the Cloud KMS [CryptoKeyVersion](https://c
      loud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.c
      ryptoKeys.cryptoKeyVersions?hl=en) used to protect this resource via
      CMEK. This field is empty if the Google Cloud product owning the
      resource does not provide key versions data to Asset Inventory. The
      first element of this field is stored in crypto_key_version.
    labels: A key-value pair of the resource's labels (v1) to their values.
    location: Location can be `global`, regional like `us-east1`, or zonal
      like `us-west1-b`.
    name: The full resource name of the resource. Example: `//compute.googleap
      is.com/projects/my_project_123/zones/zone1/instances/instance1`.
    project: Format: `projects/{PROJECT_NUMBER}`.
    projectId: The ID of the project that owns the resource.
    resourceType: Example: `compute.googleapis.com/Disk`
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A key-value pair of the resource's labels (v1) to their values.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cloudProduct = _messages.StringField(1)
    createTime = _messages.StringField(2)
    cryptoKeyVersion = _messages.StringField(3)
    cryptoKeyVersions = _messages.StringField(4, repeated=True)
    labels = _messages.MessageField('LabelsValue', 5)
    location = _messages.StringField(6)
    name = _messages.StringField(7)
    project = _messages.StringField(8)
    projectId = _messages.StringField(9)
    resourceType = _messages.StringField(10)