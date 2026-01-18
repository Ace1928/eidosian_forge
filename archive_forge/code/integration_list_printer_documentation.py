from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run.integrations import deployment_states
from googlecloudsdk.command_lib.run.integrations.formatters import base
Gets a symbol based on the latest deployment status.

  If a deployment cannot be found or the deployment is not in a 'SUCCEEDED',
  'FAILED', or 'IN_PROGRESS' state, then it should be reported as 'FAILED'.

  This would be true for integrations where the deployment never kicked off
  due to a failure.

  Args:
    status: The latest deployment status.

  Returns:
    str, the symbol to be placed in front of the integration name.
  