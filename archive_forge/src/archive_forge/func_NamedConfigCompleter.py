from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
def NamedConfigCompleter(prefix, **unused_kwargs):
    """An argcomplete completer for existing named configuration names."""
    configs = list(named_configs.ConfigurationStore.AllConfigs().keys())
    return [c for c in configs if c.startswith(prefix)]