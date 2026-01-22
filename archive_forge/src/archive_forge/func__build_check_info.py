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
def _build_check_info(app_options):
    """Get info about packages scheduled for (un)install/update"""
    handler = _AppHandler(app_options)
    messages = handler.build_check(fast=True)
    status = {'install': [], 'uninstall': [], 'update': []}
    for msg in messages:
        for key, pattern in _message_map.items():
            match = pattern.match(msg)
            if match:
                status[key].append(match.group('name'))
    return status