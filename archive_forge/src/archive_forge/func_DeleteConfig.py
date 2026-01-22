from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
@staticmethod
def DeleteConfig(config_name):
    """Creates the given configuration.

    Args:
      config_name: str, The name of the configuration to delete.

    Raises:
      NamedConfigError: If the configuration does not exist.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
    _EnsureValidConfigName(config_name, allow_reserved=False)
    paths = config.Paths()
    file_path = _FileForConfig(config_name, paths)
    if not os.path.exists(file_path):
        raise NamedConfigError('Cannot delete configuration [{0}], it does not exist.'.format(config_name))
    if config_name == _EffectiveActiveConfigName():
        raise NamedConfigError('Cannot delete configuration [{0}], it is the currently active configuration.'.format(config_name))
    if config_name == _ActiveConfigNameFromFile():
        raise NamedConfigError('Cannot delete configuration [{0}], it is currently set as the active configuration in your gcloud properties.'.format(config_name))
    try:
        os.remove(file_path)
    except (OSError, IOError) as e:
        raise NamedConfigFileAccessError('Failed to delete configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_directory), e)