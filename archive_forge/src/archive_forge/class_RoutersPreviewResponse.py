from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutersPreviewResponse(_messages.Message):
    """A RoutersPreviewResponse object.

  Fields:
    resource: Preview of given router.
  """
    resource = _messages.MessageField('Router', 1)