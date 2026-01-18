import os
import types
import inspect
from functools import wraps
from jupyter_core.paths import jupyter_config_path
from traitlets.traitlets import is_trait
from jupyter_server.services.config.manager import ConfigManager
from .traits import NotebookAppTraits
Dictionary with extension package names as keys
        and an ExtensionPackage objects as values.
        