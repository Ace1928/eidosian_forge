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
def validate_plugin_class(self, plugin_class: Type[RuntimeEnvPlugin]) -> None:
    if not issubclass(plugin_class, RuntimeEnvPlugin):
        raise RuntimeError(f'Invalid runtime env plugin class {plugin_class}. The plugin class must inherit ray._private.runtime_env.plugin.RuntimeEnvPlugin.')
    if not plugin_class.name:
        raise RuntimeError(f'No valid name in runtime env plugin {plugin_class}.')
    if plugin_class.name in self.plugins:
        raise RuntimeError(f'The name of runtime env plugin {plugin_class} conflicts with {self.plugins[plugin_class.name]}.')