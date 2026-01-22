from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1ResourceInfo(_messages.Message):
    """Describes a resource associated with this operation.

  Fields:
    permission: The resource permission required for this request.
    resourceContainer: The identifier of the parent of this resource instance.
      Must be in one of the following formats: - `projects/` - `folders/` -
      `organizations/`
    resourceLocation: The location of the resource. If not empty, the resource
      will be checked against location policy. The value must be a valid zone,
      region or multiregion. For example: "europe-west4" or "northamerica-
      northeast1-a"
    resourceName: Name of the resource. This is used for auditing purposes.
  """
    permission = _messages.StringField(1)
    resourceContainer = _messages.StringField(2)
    resourceLocation = _messages.StringField(3)
    resourceName = _messages.StringField(4)