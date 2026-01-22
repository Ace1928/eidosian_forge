from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Hl7TypesConfig(_messages.Message):
    """Root config for HL7v2 datatype definitions for a specific HL7v2 version.

  Fields:
    type: The HL7v2 type definitions.
    version: The version selectors that this config applies to. A message must
      match ALL version sources to apply.
  """
    type = _messages.MessageField('Type', 1, repeated=True)
    version = _messages.MessageField('VersionSource', 2, repeated=True)