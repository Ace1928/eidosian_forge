import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union
import tornado
from jupyterlab_server.translation_utils import translator
from traitlets import Enum
from traitlets.config import Configurable, LoggingConfigurable
from jupyterlab.commands import (
def _find_locked(self, plugins_or_extensions: List[str]) -> FrozenSet[str]:
    """Find a subset of plugins (or extensions) which are locked"""
    if self.options.lock_all:
        return set(plugins_or_extensions)
    locked_subset = set()
    extensions_with_locked_plugins = {plugin.split(':')[0] for plugin in self.options.lock_rules}
    for plugin in plugins_or_extensions:
        if ':' in plugin:
            if plugin in self.options.lock_rules:
                locked_subset.add(plugin)
        elif plugin in extensions_with_locked_plugins:
            locked_subset.add(plugin)
    return locked_subset