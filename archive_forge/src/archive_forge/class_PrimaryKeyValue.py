from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimaryKeyValue(_messages.Message):
    """Represents the primary key constraint on a table's columns.

    Fields:
      columns: Required. The columns that are composed of the primary key
        constraint.
    """
    columns = _messages.StringField(1, repeated=True)