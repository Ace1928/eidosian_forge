from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeResourcePoliciesGetRequest(_messages.Message):
    """A ComputeResourcePoliciesGetRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region for this request.
    resourcePolicy: Name of the resource policy to retrieve.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    resourcePolicy = _messages.StringField(3, required=True)