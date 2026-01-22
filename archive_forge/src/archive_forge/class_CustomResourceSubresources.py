from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceSubresources(_messages.Message):
    """CustomResourceSubresources defines the status and scale subresources for
  CustomResources.

  Fields:
    scale: Scale denotes the scale subresource for CustomResources +optional
    status: Status denotes the status subresource for CustomResources
      +optional
  """
    scale = _messages.MessageField('CustomResourceSubresourceScale', 1)
    status = _messages.MessageField('CustomResourceSubresourceStatus', 2)