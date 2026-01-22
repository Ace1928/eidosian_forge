from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomImageConfig(_messages.Message):
    """CustomImageConfig contains the information r

  Fields:
    image: The name of the image to use for this node.
    imageFamily: The name of the image family to use for this node.
    imageProject: The project containing the image to use for this node.
  """
    image = _messages.StringField(1)
    imageFamily = _messages.StringField(2)
    imageProject = _messages.StringField(3)