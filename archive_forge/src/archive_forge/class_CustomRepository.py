from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomRepository(_messages.Message):
    """Custom Go remote repository.

  Fields:
    uri: An http/https uri reference to the upstream remote repository, Must
      be the URI of a version control system. For example: https://github.com.
  """
    uri = _messages.StringField(1)