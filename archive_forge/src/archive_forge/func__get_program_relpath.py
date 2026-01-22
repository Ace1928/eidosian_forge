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
def _get_program_relpath(self, source_type: str, metadata: Dict[str, Any]) -> Optional[str]:
    if self._is_notebook_run:
        _logger.info('run is notebook based run')
        program = metadata.get('program')
        if not program:
            wandb.termwarn("Notebook 'program' path not found in metadata. See https://docs.wandb.ai/guides/launch/create-job")
        return program
    if source_type == 'artifact' or self._settings.job_source == 'artifact':
        return metadata.get('codePathLocal') or metadata.get('codePath')
    return metadata.get('codePath')