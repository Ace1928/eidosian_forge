from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourceProperties(_messages.Message):
    """The properties associated with the resource of the request.

  Fields:
    excludesDescendants: Whether an approval will exclude the descendants of
      the resource being requested.
  """
    excludesDescendants = _messages.BooleanField(1)