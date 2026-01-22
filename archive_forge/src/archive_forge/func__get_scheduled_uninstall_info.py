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
def _get_scheduled_uninstall_info(self, name) -> Optional[dict]:
    """Get information about a package that is scheduled for uninstallation"""
    target = self.app_dir / 'staging' / 'node_modules' / name / 'package.json'
    if target.exists():
        with target.open() as fid:
            return json.load(fid)
    else:
        return None