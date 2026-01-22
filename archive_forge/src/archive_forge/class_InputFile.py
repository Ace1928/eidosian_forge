from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InputFile(_messages.Message):
    """An input file.

  Fields:
    gcsSource: Google Cloud Storage file source.
    usage: Optional. Usage of the file contents. Options are
      TRAIN|VALIDATION|TEST, or UNASSIGNED (by default) for auto split.
  """
    gcsSource = _messages.MessageField('GcsInputSource', 1)
    usage = _messages.StringField(2)