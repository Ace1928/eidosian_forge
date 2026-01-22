from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Precondition(_messages.Message):
    """A precondition on a document, used for conditional operations.

  Fields:
    exists: When set to `true`, the target document must exist. When set to
      `false`, the target document must not exist.
    updateTime: When set, the target document must exist and have been last
      updated at that time. Timestamp must be microsecond aligned.
  """
    exists = _messages.BooleanField(1)
    updateTime = _messages.StringField(2)