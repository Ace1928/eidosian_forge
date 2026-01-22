from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepInstallMsi(_messages.Message):
    """Installs an MSI file.

  Fields:
    allowedExitCodes: Return codes that indicate that the software installed
      or updated successfully. Behaviour defaults to [0]
    artifactId: Required. The id of the relevant artifact in the recipe.
    flags: The flags to use when installing the MSI defaults to ["/i"] (i.e.
      the install flag).
  """
    allowedExitCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    artifactId = _messages.StringField(2)
    flags = _messages.StringField(3, repeated=True)