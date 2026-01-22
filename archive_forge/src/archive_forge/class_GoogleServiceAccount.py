from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleServiceAccount(_messages.Message):
    """Google service account

  Fields:
    accountEmail: Email address of the service account.
    subjectId: Unique identifier for the service account.
  """
    accountEmail = _messages.StringField(1)
    subjectId = _messages.StringField(2)