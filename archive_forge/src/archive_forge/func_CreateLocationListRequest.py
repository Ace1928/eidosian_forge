from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
def CreateLocationListRequest(args, messages):
    project_name = util.GetProjectResource(args.project).RelativeName()
    return messages.NotebooksProjectsLocationsListRequest(name=project_name)