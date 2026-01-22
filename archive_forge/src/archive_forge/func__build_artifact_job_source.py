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
def _build_artifact_job_source(self, program_relpath: str, metadata: Dict[str, Any]) -> Tuple[Optional[ArtifactSourceDict], Optional[str]]:
    assert isinstance(self._logged_code_artifact, dict)
    if self._is_notebook_run and (not self._is_colab_run()):
        full_program_relpath = os.path.relpath(program_relpath, os.getcwd())
        if not os.path.exists(full_program_relpath):
            if not os.path.exists(os.path.basename(program_relpath)):
                _logger.info('target path does not exist, exiting')
                wandb.termwarn('No program path found when generating artifact job source for a non-colab notebook run. See https://docs.wandb.ai/guides/launch/create-job')
                return (None, None)
            full_program_relpath = os.path.basename(program_relpath)
    else:
        full_program_relpath = program_relpath
    entrypoint = self._get_entrypoint(full_program_relpath, metadata)
    source: ArtifactSourceDict = {'entrypoint': entrypoint, 'notebook': self._is_notebook_run, 'artifact': f'wandb-artifact://_id/{self._logged_code_artifact['id']}'}
    name = self._make_job_name(self._logged_code_artifact['name'])
    return (source, name)