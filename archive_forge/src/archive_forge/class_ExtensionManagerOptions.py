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
class ExtensionManagerOptions(PluginManagerOptions):
    """Extension manager options.

    Attributes:
        allowed_extensions_uris: A list of comma-separated URIs to get the allowed extensions list
        blocked_extensions_uris: A list of comma-separated URIs to get the blocked extensions list
        listings_refresh_seconds: The interval delay in seconds to refresh the lists
        listings_tornado_options: The optional kwargs to use for the listings HTTP requests as described on https://www.tornadoweb.org/en/stable/httpclient.html#tornado.httpclient.HTTPRequest
    """
    allowed_extensions_uris: Set[str] = field(default_factory=set)
    blocked_extensions_uris: Set[str] = field(default_factory=set)
    listings_refresh_seconds: int = 60 * 60
    listings_tornado_options: dict = field(default_factory=dict)