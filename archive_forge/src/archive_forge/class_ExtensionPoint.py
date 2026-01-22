from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
class ExtensionPoint(HasTraits):
    """A simple API for connecting to a Jupyter Server extension
    point defined by metadata and importable from a Python package.
    """
    _linked = Bool(False)
    _app = Any(None, allow_none=True)
    metadata = Dict()

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

    @property
    def linked(self):
        """Has this extension point been linked to the server.

        Will pull from ExtensionApp's trait, if this point
        is an instance of ExtensionApp.
        """
        if self.app:
            return self.app._linked
        return self._linked

    @property
    def app(self):
        """If the metadata includes an `app` field"""
        return self._app

    @property
    def config(self):
        """Return any configuration provided by this extension point."""
        if self.app:
            return self.app._jupyter_server_config()
        else:
            return {}

    @property
    def module_name(self):
        """Name of the Python package module where the extension's
        _load_jupyter_server_extension can be found.
        """
        return self._module_name

    @property
    def name(self):
        """Name of the extension.

        If it's not provided in the metadata, `name` is set
        to the extensions' module name.
        """
        if self.app:
            return self.app.name
        return self.metadata.get('name', self.module_name)

    @property
    def module(self):
        """The imported module (using importlib.import_module)"""
        return self._module

    def _get_linker(self):
        """Get a linker."""
        if self.app:
            linker = self.app._link_jupyter_server_extension
        else:
            linker = getattr(self.module, '_link_jupyter_server_extension', lambda serverapp: None)
        return linker

    def _get_loader(self):
        """Get a loader."""
        loc = self.app
        if not loc:
            loc = self.module
        loader = get_loader(loc)
        return loader

    def validate(self):
        """Check that both a linker and loader exists."""
        try:
            self._get_linker()
            self._get_loader()
        except Exception:
            return False
        else:
            return True

    def link(self, serverapp):
        """Link the extension to a Jupyter ServerApp object.

        This looks for a `_link_jupyter_server_extension` function
        in the extension's module or ExtensionApp class.
        """
        if not self.linked:
            linker = self._get_linker()
            linker(serverapp)
            self._linked = True

    def load(self, serverapp):
        """Load the extension in a Jupyter ServerApp object.

        This looks for a `_load_jupyter_server_extension` function
        in the extension's module or ExtensionApp class.
        """
        loader = self._get_loader()
        return loader(serverapp)