from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsCreateRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsCreateRequest object.

  Fields:
    location: The location of the pool to create. Format:
      `locations/{location}`.
    workforcePool: A WorkforcePool resource to be passed as the request body.
    workforcePoolId: The ID to use for the pool, which becomes the final
      component of the resource name. The IDs must be a globally unique string
      of 6 to 63 lowercase letters, digits, or hyphens. It must start with a
      letter, and cannot have a trailing hyphen. The prefix `gcp-` is reserved
      for use by Google, and may not be specified.
  """
    location = _messages.StringField(1, required=True)
    workforcePool = _messages.MessageField('WorkforcePool', 2)
    workforcePoolId = _messages.StringField(3)