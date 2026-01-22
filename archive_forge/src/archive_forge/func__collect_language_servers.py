import asyncio
import os
import sys
import traceback
from typing import Dict, Text, Tuple, cast
from jupyter_core.paths import jupyter_config_path
from jupyter_server.services.config import ConfigManager
from traitlets import Bool
from traitlets import Dict as Dict_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from .constants import (
from .schema import LANGUAGE_SERVER_SPEC_MAP
from .session import LanguageServerSession
from .trait_types import LoadableCallable, Schema
from .types import (
def _collect_language_servers(self, only_installed: bool) -> KeyedLanguageServerSpecs:
    language_servers: KeyedLanguageServerSpecs = {}
    language_servers_from_config = dict(self._language_servers_from_config)
    language_servers_from_config.update(self.conf_d_language_servers)
    if self.autodetect:
        language_servers.update(self._autodetect_language_servers(only_installed=only_installed))
    language_servers.update(language_servers_from_config)
    return {key: spec for key, spec in language_servers.items() if spec.get('argv')}