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
@dataclass(frozen=True)
class PluginManagerOptions:
    """Plugin manager options.

    Attributes:
        lock_all: Whether to lock (prevent enabling/disabling) all plugins.
        lock_rules: A list of plugins or extensions that cannot be toggled.
            If extension name is provided, all its plugins will be disabled.
            The plugin names need to follow colon-separated format of `extension:plugin`.
    """
    lock_rules: FrozenSet[str] = field(default_factory=frozenset)
    lock_all: bool = False