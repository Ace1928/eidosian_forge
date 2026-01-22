from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def _load_metadata(self):
    """Import package and load metadata

        Only used if extension package is enabled
        """
    name = self.name
    try:
        self.module, self.metadata = get_metadata(name, logger=self.log)
    except ImportError as e:
        msg = f"The module '{name}' could not be found ({e}). Are you sure the extension is installed?"
        raise ExtensionModuleNotFound(msg) from None
    for m in self.metadata:
        point = ExtensionPoint(metadata=m)
        self.extension_points[point.name] = point
    return name