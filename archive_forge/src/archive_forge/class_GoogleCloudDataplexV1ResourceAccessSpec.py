from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ResourceAccessSpec(_messages.Message):
    """ResourceAccessSpec holds the access control configuration to be enforced
  on the resources, for example, Cloud Storage bucket, BigQuery dataset,
  BigQuery table.

  Fields:
    owners: Optional. The set of principals to be granted owner role on the
      resource.
    readers: Optional. The format of strings follows the pattern followed by
      IAM in the bindings. user:{email}, serviceAccount:{email} group:{email}.
      The set of principals to be granted reader role on the resource.
    writers: Optional. The set of principals to be granted writer role on the
      resource.
  """
    owners = _messages.StringField(1, repeated=True)
    readers = _messages.StringField(2, repeated=True)
    writers = _messages.StringField(3, repeated=True)