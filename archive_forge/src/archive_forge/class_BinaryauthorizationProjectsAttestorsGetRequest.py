from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationProjectsAttestorsGetRequest(_messages.Message):
    """A BinaryauthorizationProjectsAttestorsGetRequest object.

  Fields:
    name: Required. The name of the attestor to retrieve, in the format
      `projects/*/attestors/*`.
  """
    name = _messages.StringField(1, required=True)