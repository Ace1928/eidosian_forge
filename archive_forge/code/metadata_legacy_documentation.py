import logging
import os
from pip._internal.build_env import BuildEnvironment
from pip._internal.cli.spinners import open_spinner
from pip._internal.exceptions import (
from pip._internal.utils.setuptools_build import make_setuptools_egg_info_args
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory
Generate metadata using setup.py-based defacto mechanisms.

    Returns the generated metadata directory.
    