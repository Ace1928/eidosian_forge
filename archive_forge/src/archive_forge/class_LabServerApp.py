from glob import glob
from os.path import relpath
from typing import Any
from jupyter_server.extension.application import ExtensionApp, ExtensionAppJinjaMixin
from jupyter_server.utils import url_path_join as ujoin
from traitlets import Dict, Integer, Unicode, observe
from ._version import __version__
from .handlers import LabConfig, add_handlers
class LabServerApp(ExtensionAppJinjaMixin, LabConfig, ExtensionApp):
    """A Lab Server Application that runs out-of-the-box"""
    name = 'jupyterlab_server'
    extension_url = '/lab'
    app_name = 'JupyterLab Server Application'
    file_url_prefix = '/lab/tree'

    @property
    def app_namespace(self) -> str:
        return self.name
    default_url = Unicode('/lab', help='The default URL to redirect to from `/`')
    load_other_extensions = True
    app_version = Unicode('', help='The version of the application.').tag(default=__version__)
    blacklist_uris = Unicode('', config=True, help='Deprecated, use `LabServerApp.blocked_extensions_uris`')
    blocked_extensions_uris = Unicode('', config=True, help='\n        A list of comma-separated URIs to get the blocked extensions list\n\n        .. versionchanged:: 2.0.0\n            `LabServerApp.blacklist_uris` renamed to `blocked_extensions_uris`\n        ')
    whitelist_uris = Unicode('', config=True, help='Deprecated, use `LabServerApp.allowed_extensions_uris`')
    allowed_extensions_uris = Unicode('', config=True, help='\n        "A list of comma-separated URIs to get the allowed extensions list\n\n        .. versionchanged:: 2.0.0\n            `LabServerApp.whitetlist_uris` renamed to `allowed_extensions_uris`\n        ')
    listings_refresh_seconds = Integer(60 * 60, config=True, help='The interval delay in seconds to refresh the lists')
    listings_request_options = Dict({}, config=True, help='The optional kwargs to use for the listings HTTP requests             as described on https://2.python-requests.org/en/v2.7.0/api/#requests.request')
    _deprecated_aliases = {'blacklist_uris': ('blocked_extensions_uris', '1.2'), 'whitelist_uris': ('allowed_extensions_uris', '1.2')}

    @observe(*list(_deprecated_aliases))
    def _deprecated_trait(self, change: Any) -> None:
        """observer for deprecated traits"""
        old_attr = change.name
        new_attr, version = self._deprecated_aliases.get(old_attr)
        new_value = getattr(self, new_attr)
        if new_value != change.new:
            self.log.warning('%s.%s is deprecated in JupyterLab %s, use %s.%s instead', self.__class__.__name__, old_attr, version, self.__class__.__name__, new_attr)
            setattr(self, new_attr, change.new)

    def initialize_settings(self) -> None:
        """Initialize the settings:

        set the static files as immutable, since they should have all hashed name.
        """
        immutable_cache = set(self.settings.get('static_immutable_cache', []))
        immutable_cache.add(self.static_url_prefix)
        for extension_path in self.labextensions_path + self.extra_labextensions_path:
            extensions_url = [ujoin(self.labextensions_url, relpath(path, extension_path)) for path in glob(f'{extension_path}/**/static', recursive=True)]
            immutable_cache.update(extensions_url)
        self.settings.update({'static_immutable_cache': list(immutable_cache)})

    def initialize_templates(self) -> None:
        """Initialize templates."""
        self.static_paths = [self.static_dir]
        self.template_paths = [self.templates_dir]

    def initialize_handlers(self) -> None:
        """Initialize handlers."""
        add_handlers(self.handlers, self)