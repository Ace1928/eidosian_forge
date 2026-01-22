from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateExclusivityManifestResponse(_messages.Message):
    """The response of the exclusivity artifacts manifests for the client to
  apply.

  Fields:
    crManifest: The YAML manifest of the membership CR to apply if a new
      version of the CR is available. Empty if no update needs to be applied.
    crdManifest: The YAML manifest of the membership CRD to apply if a newer
      version of the CRD is available. Empty if no update needs to be applied.
  """
    crManifest = _messages.StringField(1)
    crdManifest = _messages.StringField(2)