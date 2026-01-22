from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import walker
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.cache import resource_cache
import six
class _CompleterModuleGenerator(walker.Walker):
    """Constructs a CLI command dict tree."""

    def __init__(self, cli):
        super(_CompleterModuleGenerator, self).__init__(cli)
        self._modules_dict = {}

    def Visit(self, command, parent, is_group):
        """Visits each command in the CLI command tree to construct the module list.

    Args:
      command: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if command is a group, otherwise its is a command.

    Returns:
      The subtree module list.
    """

        def _ActionKey(action):
            return action.__repr__()
        args = command.ai
        for arg in sorted(args.flag_args + args.positional_args, key=_ActionKey):
            try:
                completer_class = arg.completer
            except AttributeError:
                continue
            collection = None
            api_version = None
            if isinstance(completer_class, parser_completer.ArgumentCompleter):
                completer_class = completer_class.completer_class
            module_path = module_util.GetModulePath(completer_class)
            if isinstance(completer_class, type):
                try:
                    completer = completer_class()
                    try:
                        collection = completer.collection
                    except AttributeError:
                        pass
                    try:
                        api_version = completer.api_version
                    except AttributeError:
                        pass
                except (apis_util.UnknownAPIError, resources.InvalidCollectionException) as e:
                    collection = 'ERROR: {}'.format(e)
            if arg.option_strings:
                name = arg.option_strings[0]
            else:
                name = arg.dest.replace('_', '-')
            module = self._modules_dict.get(module_path)
            if not module:
                module = _CompleterModule(module_path=module_path, collection=collection, api_version=api_version, completer_type=_GetCompleterType(completer_class))
                self._modules_dict[module_path] = module
            command_path = ' '.join(command.GetPath())
            attachment = module._attachments_dict.get(command_path)
            if not attachment:
                attachment = _CompleterAttachment(command_path)
                module._attachments_dict[command_path] = attachment
                module.attachments.append(attachment)
            attachment.arguments.append(name)
        return self._modules_dict