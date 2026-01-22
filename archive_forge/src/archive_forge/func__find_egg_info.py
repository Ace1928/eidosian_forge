import logging
import os
from pip._internal.build_env import BuildEnvironment
from pip._internal.cli.spinners import open_spinner
from pip._internal.exceptions import (
from pip._internal.utils.setuptools_build import make_setuptools_egg_info_args
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory
def _find_egg_info(directory: str) -> str:
    """Find an .egg-info subdirectory in `directory`."""
    filenames = [f for f in os.listdir(directory) if f.endswith('.egg-info')]
    if not filenames:
        raise InstallationError(f'No .egg-info directory found in {directory}')
    if len(filenames) > 1:
        raise InstallationError('More than one .egg-info directory found in {}'.format(directory))
    return os.path.join(directory, filenames[0])