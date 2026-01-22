from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginHeaderActionAddHeader(_messages.Message):
    """Describes a header to add.

  Fields:
    headerName: Required. The name of the header to add.
    headerValue: Required. The value of the header to add.
    replace: Optional. Specifies whether to replace all existing headers with
      the same name. By default, added header values are appended to the
      response or request headers with the same field names. The added values
      are separated by commas. To overwrite existing values, set `replace` to
      `true`.
  """
    headerName = _messages.StringField(1)
    headerValue = _messages.StringField(2)
    replace = _messages.BooleanField(3)