from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsPatchRequest(_messages.Message):
    """A AppengineAppsPatchRequest object.

  Fields:
    application: A Application resource to be passed as the request body.
    name: Name of the Application resource to update. Example: apps/myapp.
    updateMask: Required. Standard field mask for the set of fields to be
      updated.
  """
    application = _messages.MessageField('Application', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)