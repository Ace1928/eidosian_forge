from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PatientId(_messages.Message):
    """A patient identifier and associated type.

  Fields:
    type: ID type. For example, MRN or NHS.
    value: The patient's unique identifier.
  """
    type = _messages.StringField(1)
    value = _messages.StringField(2)