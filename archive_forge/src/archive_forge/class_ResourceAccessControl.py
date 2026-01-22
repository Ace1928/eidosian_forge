from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceAccessControl(_messages.Message):
    """The access controls set on the resource.

  Fields:
    gcpIamPolicy: The GCP IAM Policy to set on the resource.
  """
    gcpIamPolicy = _messages.StringField(1)