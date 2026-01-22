from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollBareMetalNodePoolRequest(_messages.Message):
    """Message for enrolling an existing bare metal node pool to the GKE on-
  prem API.

  Fields:
    bareMetalNodePoolId: User provided OnePlatform identifier that is used as
      part of the resource name. (https://tools.ietf.org/html/rfc1123) format.
    validateOnly: If set, only validate the request, but do not actually
      enroll the node pool.
  """
    bareMetalNodePoolId = _messages.StringField(1)
    validateOnly = _messages.BooleanField(2)