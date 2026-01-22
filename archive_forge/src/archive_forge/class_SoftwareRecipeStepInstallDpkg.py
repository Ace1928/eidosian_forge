from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepInstallDpkg(_messages.Message):
    """Installs a deb via dpkg.

  Fields:
    artifactId: Required. The id of the relevant artifact in the recipe.
  """
    artifactId = _messages.StringField(1)