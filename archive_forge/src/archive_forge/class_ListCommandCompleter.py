from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.api_lib.util import resource_search
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import completion_cache
from googlecloudsdk.core.cache import resource_cache
import six
class ListCommandCompleter(ResourceCompleter):
    """A parameterized completer that uses a gcloud list command for updates.

  Attributes:
    list_command: The gcloud list command that returns the list of current
      resource URIs.
    flags: The resource parameter flags that are referenced by list_command.
    parse_output: The completion items are written to the list_command standard
      output, one per line, if True. Otherwise the list_command return value is
      the list of items.
  """

    def __init__(self, list_command=None, flags=None, parse_output=False, **kwargs):
        self._list_command = list_command
        self._flags = flags or []
        self._parse_output = parse_output
        super(ListCommandCompleter, self).__init__(**kwargs)

    def GetListCommand(self, parameter_info):
        """Returns the list command argv given parameter_info."""

        def _FlagName(flag):
            return flag.split('=')[0]
        list_command = self._list_command.split()
        flags = {_FlagName(f) for f in list_command if f.startswith('--')}
        if '--quiet' not in flags:
            flags.add('--quiet')
            list_command.append('--quiet')
        if '--uri' in flags and '--format' not in flags:
            flags.add('--format')
            list_command.append('--format=disable')
        for name in self._flags + [parameter.name for parameter in self.parameters] + parameter_info.GetAdditionalParams():
            flag = parameter_info.GetFlag(name, check_properties=False, for_update=True)
            if flag:
                flag_name = _FlagName(flag)
                if flag_name not in flags:
                    flags.add(flag_name)
                    list_command.append(flag)
        return list_command

    def GetAllItems(self, command, parameter_info):
        """Runs command and returns the list of completion items."""
        try:
            if not self._parse_output:
                return parameter_info.Execute(command)
            log_out = log.out
            out = io.StringIO()
            log.out = out
            parameter_info.Execute(command)
            return out.getvalue().rstrip('\n').split('\n')
        finally:
            if self._parse_output:
                log.out = log_out

    def Update(self, parameter_info, aggregations):
        """Returns the current list of parsed resources from list_command."""
        command = self.GetListCommand(parameter_info)
        for parameter in aggregations:
            flag = parameter_info.GetFlag(parameter.name, parameter.value, for_update=True)
            if flag and flag not in command:
                command.append(flag)
        log.info('cache update command: %s' % ' '.join(command))
        try:
            items = list(self.GetAllItems(command, parameter_info) or [])
        except (Exception, SystemExit) as e:
            if properties.VALUES.core.print_completion_tracebacks.GetBool():
                raise
            log.info(six.text_type(e).rstrip())
            try:
                raise type(e)('Update command [{}]: {}'.format(' '.join(command), six.text_type(e).rstrip()))
            except TypeError:
                raise e
        return [self.StringToRow(item, parameter_info) for item in items]