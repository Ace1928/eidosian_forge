from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import resources
class IamRolesCompleter(completers.ListCommandCompleter):
    """An IAM role completer for a resource argument.

  The Complete() method override bypasses the completion cache.

  Attributes:
    _resource_dest: The argparse Namespace dest string for the resource
      argument that has the roles.
    _resource_collection: The resource argument collection.
  """

    def __init__(self, resource_dest=None, resource_collection=None, **kwargs):
        super(IamRolesCompleter, self).__init__(**kwargs)
        self._resource_dest = resource_dest
        self._resource_collection = resource_collection

    def GetListCommand(self, parameter_info):
        resource_ref = resources.REGISTRY.Parse(parameter_info.GetValue(self._resource_dest), collection=self._resource_collection, default_resolver=parameter_info.GetValue)
        resource_uri = resource_ref.SelfLink()
        return ['beta', 'iam', 'list-grantable-roles', '--quiet', '--flatten=name', '--format=disable', resource_uri]

    def Complete(self, prefix, parameter_info):
        """Bypasses the cache and returns completions matching prefix."""
        command = self.GetListCommand(parameter_info)
        items = self.GetAllItems(command, parameter_info)
        return [item for item in items or [] if item is not None and item.startswith(prefix)]