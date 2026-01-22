from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util import completers
class ProjectsIamRolesCompleter(iam_completers.IamRolesCompleter):
    """IAM Roles Completer."""

    def __init__(self, **kwargs):
        super(ProjectsIamRolesCompleter, self).__init__(resource_collection='cloudresourcemanager.projects', resource_dest='project_id', **kwargs)