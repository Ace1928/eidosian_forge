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
class ActivePropertiesFile(object):
    """An interface for loading and caching the active properties from file."""
    _PROPERTIES = None
    _LOCK = threading.RLock()

    @staticmethod
    def Load():
        """Loads the set of active properties from file.

    This includes both the installation configuration as well as the currently
    active configuration file.

    Returns:
      properties_file.PropertiesFile, The CloudSDK properties.
    """
        ActivePropertiesFile._LOCK.acquire()
        try:
            if not ActivePropertiesFile._PROPERTIES:
                ActivePropertiesFile._PROPERTIES = properties_file.PropertiesFile([config.Paths().installation_properties_path, ActiveConfig(force_create=False).file_path])
        finally:
            ActivePropertiesFile._LOCK.release()
        return ActivePropertiesFile._PROPERTIES

    @staticmethod
    def Invalidate(mark_changed=False):
        """Invalidate the cached property values.

    Args:
      mark_changed: bool, True if we are invalidating because we persisted
        a change to the installation config, the active configuration, or
        changed the active configuration. If so, the config sentinel is touched.
    """
        ActivePropertiesFile._PROPERTIES = None
        if mark_changed:
            file_utils.WriteFileContents(config.Paths().config_sentinel_file, '')