from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SBOMReferenceNote(_messages.Message):
    """The note representing an SBOM reference.

  Fields:
    format: The format that SBOM takes. E.g. may be spdx, cyclonedx, etc...
    version: The version of the format that the SBOM takes. E.g. if the format
      is spdx, the version may be 2.3.
  """
    format = _messages.StringField(1)
    version = _messages.StringField(2)