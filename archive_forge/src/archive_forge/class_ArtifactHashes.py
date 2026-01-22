from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactHashes(_messages.Message):
    """Defines a hash object for use in Materials and Products.

  Fields:
    sha256: A string attribute.
  """
    sha256 = _messages.StringField(1)