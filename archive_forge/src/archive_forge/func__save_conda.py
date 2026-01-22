import datetime
import glob
import json
import logging
import os
import subprocess
import sys
from shutil import copyfile
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.filenames import CONDA_ENVIRONMENTS_FNAME, DIFF_FNAME, METADATA_FNAME
from wandb.sdk.lib.gitlib import GitRepo
from .assets.interfaces import Interface
def _save_conda(self) -> None:
    current_shell_is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not current_shell_is_conda:
        return None
    logger.debug('Saving list of conda packages installed into the current environment')
    try:
        with open(os.path.join(self.settings.files_dir, CONDA_ENVIRONMENTS_FNAME), 'w') as f:
            subprocess.call(['conda', 'env', 'export'], stdout=f, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.exception(f'Error saving conda packages: {e}')
    logger.debug('Saving conda packages done')