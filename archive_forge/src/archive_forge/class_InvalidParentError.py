from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidParentError(Error):
    """An error representing an invalid CRM parent."""

    def __init__(self, bad_parent_arg: str):
        super(Error, self).__init__(f'"{bad_parent_arg}" is not a valid parent. The parent name should begin with "organizations/", "projects/", or "folders/".')