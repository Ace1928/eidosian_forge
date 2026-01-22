from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util import completers
class ProjectCompleter(completers.ResourceParamCompleter):
    """The project completer."""

    def __init__(self, **kwargs):
        super(ProjectCompleter, self).__init__(collection='cloudresourcemanager.projects', list_command='projects list --uri', param='projectId', **kwargs)