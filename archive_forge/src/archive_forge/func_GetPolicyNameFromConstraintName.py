from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.org_policies import exceptions
def GetPolicyNameFromConstraintName(constraint_name):
    """Returns the associated policy name for the specified constraint name.

  A policy name has the following syntax:
  [organizations|folders|projects]/{resource_id}/policies/{constraint_name}.

  Args:
    constraint_name: The name of the constraint. A constraint name has the
      following syntax:
        [organizations|folders|projects]/{resource_id}/constraints/{constraint_name}.
  """
    constraint_name_tokens = _GetConstraintNameTokens(constraint_name)
    return '{}/{}/policies/{}'.format(constraint_name_tokens[0], constraint_name_tokens[1], constraint_name_tokens[3])