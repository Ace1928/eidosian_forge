import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
def add_plugin(self, plugin: RuntimeEnvPlugin) -> None:
    """Add a plugin to the manager and create a URI cache for it.

        Args:
            plugin: The class instance of the plugin.
        """
    plugin_class = type(plugin)
    self.validate_plugin_class(plugin_class)
    self.validate_priority(plugin_class.priority)
    self.plugins[plugin_class.name] = PluginSetupContext(plugin_class.name, plugin, plugin_class.priority, self.create_uri_cache_for_plugin(plugin))