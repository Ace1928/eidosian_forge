from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterStatusResponse(_messages.Message):
    """A RouterStatusResponse object.

  Fields:
    kind: Type of resource.
    result: A RouterStatus attribute.
  """
    kind = _messages.StringField(1, default='compute#routerStatusResponse')
    result = _messages.MessageField('RouterStatus', 2)