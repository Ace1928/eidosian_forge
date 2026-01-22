import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def _is_artifact_reference(self) -> bool:
    return self.ref is not None and urlparse(self.ref).scheme == 'wandb-artifact'