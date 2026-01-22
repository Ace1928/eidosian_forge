import json
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.lib.filenames import DIFF_FNAME, METADATA_FNAME, REQUIREMENTS_FNAME
from wandb.util import make_artifact_name_safe
from .settings_static import SettingsStatic
def _handle_metadata_file(self) -> Optional[Dict]:
    if os.path.exists(os.path.join(self._settings.files_dir, METADATA_FNAME)):
        with open(os.path.join(self._settings.files_dir, METADATA_FNAME)) as f:
            metadata: Dict = json.load(f)
        return metadata
    return None