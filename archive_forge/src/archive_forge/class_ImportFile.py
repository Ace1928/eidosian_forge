from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportFile(_messages.Message):
    """A ImportFile object.

  Fields:
    content: The contents of the file.
    name: The name of the file.
  """
    content = _messages.StringField(1)
    name = _messages.StringField(2)