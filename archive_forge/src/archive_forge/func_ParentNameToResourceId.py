from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
def ParentNameToResourceId(parent_name, api_version=DEFAULT_API_VERSION):
    messages = projects_util.GetMessages(api_version)
    if not parent_name:
        return None
    elif parent_name.startswith('folders/'):
        return messages.ResourceId(id=folders.FolderNameToId(parent_name), type='folder')
    elif parent_name.startswith('organizations/'):
        return messages.ResourceId(id=parent_name[len('organizations/'):], type='organization')