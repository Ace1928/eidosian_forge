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
def ActivateConfig(config_name):
    """Activates an existing named configuration.

    Args:
      config_name: str, The name of the configuration to activate.

    Raises:
      NamedConfigError: If the configuration does not exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
    _EnsureValidConfigName(config_name, allow_reserved=True)
    paths = config.Paths()
    file_path = _FileForConfig(config_name, paths)
    if file_path and (not os.path.exists(file_path)):
        raise NamedConfigError('Cannot activate configuration [{0}], it does not exist.'.format(config_name))
    try:
        file_utils.WriteFileContents(paths.named_config_activator_path, config_name)
    except file_utils.Error as e:
        raise NamedConfigFileAccessError('Failed to activate configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_activator_path), e)
    ActivePropertiesFile.Invalidate(mark_changed=True)