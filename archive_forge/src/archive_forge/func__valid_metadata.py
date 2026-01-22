from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
@validate_trait('metadata')
def _valid_metadata(self, proposed):
    """Validate metadata."""
    metadata = proposed['value']
    try:
        self._module_name = metadata['module']
    except KeyError:
        msg = "There is no 'module' key in the extension's metadata packet."
        raise ExtensionMetadataError(msg) from None
    try:
        self._module = importlib.import_module(self._module_name)
    except ImportError:
        msg = f"The submodule '{self._module_name}' could not be found. Are you sure the extension is installed?"
        raise ExtensionModuleNotFound(msg) from None
    if 'app' in metadata:
        self._app = metadata['app']()
    return metadata