from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1CmekConfig(_messages.Message):
    """The CMEK (Customer Managed Encryption Key) configuration for a Firestore
  database. If not present, the database is secured by the default Google
  encryption key.

  Fields:
    activeKeyVersion: Output only. Currently in-use [KMS key
      versions](https://cloud.google.com/kms/docs/resource-
      hierarchy#key_versions). During [key
      rotation](https://cloud.google.com/kms/docs/key-rotation), there can be
      multiple in-use key versions. The expected format is `projects/{project_
      id}/locations/{kms_location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}
      /cryptoKeyVersions/{key_version}`.
    kmsKeyName: Required. Only keys in the same location as this database are
      allowed to be used for encryption. For Firestore's nam5 multi-region,
      this corresponds to Cloud KMS multi-region us. For Firestore's eur3
      multi-region, this corresponds to Cloud KMS multi-region europe. See
      https://cloud.google.com/kms/docs/locations. The expected format is `pro
      jects/{project_id}/locations/{kms_location}/keyRings/{key_ring}/cryptoKe
      ys/{crypto_key}`.
  """
    activeKeyVersion = _messages.StringField(1, repeated=True)
    kmsKeyName = _messages.StringField(2)