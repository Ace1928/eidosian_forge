from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepExecFile(_messages.Message):
    """Executes an artifact or local file.

  Fields:
    allowedExitCodes: Defaults to [0]. A list of possible return values that
      the program can return to indicate a success.
    args: Arguments to be passed to the provided executable.
    artifactId: The id of the relevant artifact in the recipe.
    localPath: The absolute path of the file on the local filesystem.
  """
    allowedExitCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    args = _messages.StringField(2, repeated=True)
    artifactId = _messages.StringField(3)
    localPath = _messages.StringField(4)