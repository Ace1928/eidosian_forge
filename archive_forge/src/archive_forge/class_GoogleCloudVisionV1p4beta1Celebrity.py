from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1Celebrity(_messages.Message):
    """A Celebrity is a group of Faces with an identity.

  Fields:
    description: The Celebrity's description.
    displayName: The Celebrity's display name.
    name: The resource name of the preloaded Celebrity. Has the format
      `builtin/{mid}`.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)