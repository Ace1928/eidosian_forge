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
def _save_code(self) -> None:
    logger.debug('Saving code')
    if not self.settings.program_relpath:
        logger.warning('unable to save code -- program entry not found')
        return None
    root: str = self.git.root or os.getcwd()
    program_relative: str = self.settings.program_relpath
    filesystem.mkdir_exists_ok(os.path.join(self.settings.files_dir, 'code', os.path.dirname(program_relative)))
    program_absolute = os.path.join(root, program_relative)
    if not os.path.exists(program_absolute):
        logger.warning("unable to save code -- can't find %s" % program_absolute)
        return None
    saved_program = os.path.join(self.settings.files_dir, 'code', program_relative)
    self.saved_program = program_relative
    if not os.path.exists(saved_program):
        copyfile(program_absolute, saved_program)
    logger.debug('Saving code done')