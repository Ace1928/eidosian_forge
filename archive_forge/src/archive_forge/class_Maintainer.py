from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Maintainer(_messages.Message):
    """A Maintainer object.

  Fields:
    email: A string attribute.
    kind: A string attribute.
    name: A string attribute.
    url: A string attribute.
  """
    email = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3)
    url = _messages.StringField(4)