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
def _autodetect_language_servers(self, only_installed: bool):
    _entry_points = None
    try:
        _entry_points = entry_points(group=EP_SPEC_V1)
    except Exception:
        self.log.exception('Failed to load entry_points')
    skipped_servers = []
    for ep in _entry_points or []:
        try:
            spec_finder: SpecMaker = ep.load()
        except Exception as err:
            self.log.warning(_('Failed to load language server spec finder `{}`: \n{}').format(ep.name, err))
            continue
        try:
            if only_installed:
                if hasattr(spec_finder, 'is_installed'):
                    spec_finder_from_base = cast(SpecBase, spec_finder)
                    if not spec_finder_from_base.is_installed(self):
                        skipped_servers.append(ep.name)
                        continue
            specs = spec_finder(self) or {}
        except Exception as err:
            self.log.warning(_('Failed to fetch commands from language server spec finder `{}`:\n{}').format(ep.name, err))
            traceback.print_exc()
            continue
        errors = list(LANGUAGE_SERVER_SPEC_MAP.iter_errors(specs))
        if errors:
            self.log.warning(_('Failed to validate commands from language server spec finder `{}`:\n{}').format(ep.name, errors))
            continue
        for key, spec in specs.items():
            yield (key, spec)
    if skipped_servers:
        self.log.info(_('Skipped non-installed server(s): {}').format(', '.join(skipped_servers)))