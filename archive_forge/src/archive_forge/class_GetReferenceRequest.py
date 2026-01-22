from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetReferenceRequest(_messages.Message):
    """The GetReferenceRequest request.

  Fields:
    name: Required. Full resource name of the reference, in the following
      format:
      `//{target_service}/{target_resource}/references/{reference_id}`. For
      example: `//targetservice.googleapis.com/projects/{my-
      project}/locations/{location}/instances/{my-instance}/references/{xyz}`.
  """
    name = _messages.StringField(1)