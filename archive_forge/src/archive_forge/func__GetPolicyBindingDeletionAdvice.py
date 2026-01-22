from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetPolicyBindingDeletionAdvice(minimal_roles):
    """Returns advice for policy binding deletion.

  Args:
    minimal_roles: A string list of minimal recommended roles.

  Returns: A string advice on safe deletion.
  """
    if minimal_roles:
        return _POLICY_BINDING_REPLACE_ADVICE.format('' if len(minimal_roles) <= 1 else 's', ', '.join(minimal_roles))
    else:
        return _POLICY_BINDING_DELETE_ADVICE