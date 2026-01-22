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
def _handle_server_artifact(self, res: Optional[Dict], artifact: 'ArtifactRecord') -> None:
    if artifact.type == 'job' and res is not None:
        try:
            if res['artifactSequence']['latestArtifact'] is None:
                self._job_version_alias = 'v0'
            elif res['artifactSequence']['latestArtifact']['id'] == res['id']:
                self._job_version_alias = f'v{res['artifactSequence']['latestArtifact']['versionIndex']}'
            else:
                self._job_version_alias = f'v{res['artifactSequence']['latestArtifact']['versionIndex'] + 1}'
            self._job_seq_id = res['artifactSequence']['id']
        except KeyError as e:
            _logger.info(f'Malformed response from ArtifactSaver.save {e}')
    if artifact.type == 'code' and res is not None:
        self._logged_code_artifact = ArtifactInfoForJob({'id': res['id'], 'name': artifact.name})