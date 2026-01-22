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
class PluginSetupContext:

    def __init__(self, name: str, class_instance: RuntimeEnvPlugin, priority: int, uri_cache: URICache):
        self.name = name
        self.class_instance = class_instance
        self.priority = priority
        self.uri_cache = uri_cache