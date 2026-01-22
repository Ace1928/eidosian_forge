from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetExecutionCreatedMessage(release_track, execution):
    """Returns a user message with execution details when running a job."""
    msg = '\nView details about this execution by running:\ngcloud{release_track} run jobs executions describe {execution_name}'.format(release_track=' {}'.format(release_track.prefix) if release_track.prefix is not None else '', execution_name=execution.name)
    if execution.status and execution.status.logUri:
        msg += '\n\nOr visit ' + _GetExecutionUiLink(execution)
    return msg