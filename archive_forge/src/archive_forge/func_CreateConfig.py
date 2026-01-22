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
def CreateConfig(config_name):
    """Creates a configuration with the given name.

    Args:
      config_name: str, The name of the configuration to create.

    Returns:
      Configuration, The configuration that was just created.

    Raises:
      NamedConfigError: If the configuration already exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
    _EnsureValidConfigName(config_name, allow_reserved=False)
    paths = config.Paths()
    file_path = _FileForConfig(config_name, paths)
    if os.path.exists(file_path):
        raise NamedConfigError('Cannot create configuration [{0}], it already exists.'.format(config_name))
    try:
        file_utils.MakeDir(paths.named_config_directory)
        file_utils.WriteFileContents(file_path, '')
    except file_utils.Error as e:
        raise NamedConfigFileAccessError('Failed to create configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_directory), e)
    return Configuration(config_name, is_active=False)