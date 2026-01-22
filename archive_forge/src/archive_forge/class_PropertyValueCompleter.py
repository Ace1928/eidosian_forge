from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
class PropertyValueCompleter(completers.Converter):
    """A completer for a specific property value.

  The property value to be completed is not known until completion time.
  """

    def Complete(self, prefix, parameter_info):
        properties.VALUES.core.print_completion_tracebacks.Set(True)
        prop_name = parameter_info.GetValue('property')
        if not prop_name:
            return None
        prop = properties.FromString(prop_name)
        if not prop:
            return None
        if prop.choices:
            return [c for c in prop.choices if c.startswith(prefix)]
        if prop.completer:
            completer_class = module_util.ImportModule(prop.completer)
            completer = completer_class(cache=self.cache)
            return completer.Complete(prefix, parameter_info)
        return None

    def Update(self, parameter_info=None, aggregations=None):
        """No completion cache for properties."""
        del parameter_info, aggregations