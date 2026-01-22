from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def GetNonInteractiveErrorMessage():
    """Returns useful instructions when running non-interactive.

  Certain fingerprinting modules require interactive functionality.  It isn't
  always obvious why gcloud is running in non-interactive mode (e.g. when
  "disable_prompts" is set) so this returns an appropriate addition to the
  error message in these circumstances.

  Returns:
    (str) The appropriate error message snippet.
  """
    if properties.VALUES.core.disable_prompts.GetBool():
        return ' ' + _PROMPTS_DISABLED_ERROR_MESSAGE
    else:
        return ''