from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginOverrideAction(_messages.Message):
    """Defines how requests and responses can be manipulated on cache fill to
  this origin.

  Fields:
    headerAction: Optional. The header actions, including adding and removing
      headers, for requests handled by this origin.
    urlRewrite: Optional. The URL rewrite configuration for requests that are
      handled by this origin.
  """
    headerAction = _messages.MessageField('OriginHeaderAction', 1)
    urlRewrite = _messages.MessageField('OriginUrlRewrite', 2)