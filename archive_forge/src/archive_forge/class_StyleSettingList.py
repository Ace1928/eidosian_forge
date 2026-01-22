from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class StyleSettingList(_messages.Message):
    """Represents a list of styles for a given table.

  Fields:
    items: All requested style settings.
    kind: Type name: in this case, a list of style settings.
    nextPageToken: Token used to access the next page of this result. No token
      is displayed if there are no more pages left.
    totalItems: Total number of styles for the table.
  """
    items = _messages.MessageField('StyleSetting', 1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#styleSettingList')
    nextPageToken = _messages.StringField(3)
    totalItems = _messages.IntegerField(4, variant=_messages.Variant.INT32)