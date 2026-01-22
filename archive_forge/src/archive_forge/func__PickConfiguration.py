from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib import init_util
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import network_diagnostics
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _PickConfiguration(self):
    """Allows user to re-initialize, create or pick new configuration.

    Returns:
      Configuration name or None.
    """
    configs = named_configs.ConfigurationStore.AllConfigs()
    active_config = named_configs.ConfigurationStore.ActiveConfig()
    if not configs or active_config.name not in configs:
        named_configs.ConfigurationStore.CreateConfig(active_config.name)
        active_config.Activate()
        return active_config.name
    if len(configs) == 1:
        default_config = configs.get(named_configs.DEFAULT_CONFIG_NAME, None)
        if default_config and (not default_config.GetProperties()):
            default_config.Activate()
            return default_config.name
    choices = []
    log.status.write('Settings from your current configuration [{0}] are:\n'.format(active_config.name))
    log.status.flush()
    log.status.write(yaml.dump(properties.VALUES.AllValues()))
    log.out.flush()
    log.status.write('\n')
    log.status.flush()
    choices.append('Re-initialize this configuration [{0}] with new settings '.format(active_config.name))
    choices.append('Create a new configuration')
    config_choices = [name for name, c in sorted(six.iteritems(configs)) if not c.is_active]
    choices.extend(('Switch to and re-initialize existing configuration: [{0}]'.format(name) for name in config_choices))
    idx = console_io.PromptChoice(choices, message='Pick configuration to use:')
    if idx is None:
        return None
    if idx == 0:
        self._CleanCurrentConfiguration()
        return active_config.name
    if idx == 1:
        return self._CreateConfiguration()
    config_name = config_choices[idx - 2]
    named_configs.ConfigurationStore.ActivateConfig(config_name)
    return config_name