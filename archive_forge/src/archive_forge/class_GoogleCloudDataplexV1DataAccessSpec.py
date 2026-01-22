from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataAccessSpec(_messages.Message):
    """DataAccessSpec holds the access control configuration to be enforced on
  data stored within resources (eg: rows, columns in BigQuery Tables). When
  associated with data, the data is only accessible to principals explicitly
  granted access through the DataAccessSpec. Principals with access to the
  containing resource are not implicitly granted access.

  Fields:
    readers: Optional. The format of strings follows the pattern followed by
      IAM in the bindings. user:{email}, serviceAccount:{email} group:{email}.
      The set of principals to be granted reader role on data stored within
      resources.
  """
    readers = _messages.StringField(1, repeated=True)